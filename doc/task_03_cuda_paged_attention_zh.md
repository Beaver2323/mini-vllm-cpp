# 开发任务 03：CUDA PagedAttention Decode

**状态：已完成独立 FP32 Kernel 基线。**

## 任务目标

这一阶段把 CPU PagedAttention 的核心计算搬到 GPU，并保留 vLLM 风格的分页接口。
实现范围是独立的 FP32 Decode Kernel：输入当前 Token 的 Q/K/V、请求页表和上下文长度，
先把新 K/V 写入缓存，再让 Q 读取该请求的全部历史 K/V。它尚未接入完整 GPT-2
ModelRunner，因此本任务报告的是 Kernel 结果，不是模型端到端吞吐。后续任务 04 已完成
模型接入，端到端结果见 `task_04_gpu_model_runner_zh.md`。

本任务新增约 881 行 C++/CUDA：核心接口和 Kernel 218 行，独立正确性测试 319 行，
Benchmark 344 行。真正需要在面试中讲清楚的是 218 行核心实现；测试和 Benchmark
占多数，因为性能项目必须证明结果正确且可复现。

## 数据布局

```text
Q / new K / new V / output
  [batch, num_heads, head_size]

K Cache / V Cache
  [num_pages, num_layers, num_heads, page_size=16, head_size]

Block Table
  [batch, max_blocks_per_sequence]

Context Length
  [batch]
```

逻辑 Token 位置 `t` 的地址转换为：

```text
logical_page  = t / 16
offset_in_page = t % 16
physical_page = block_table[request][logical_page]
```

Attention 只依赖页表，不要求一个请求的物理页连续。这正是分页 KV Cache 与普通连续
Tensor 的关键区别。

## Kernel 如何工作

入口 `paged_attention_decode()` 连续启动两个 Kernel：

1. `write_kv_cache_kernel`：按最后一个 Token 的逻辑位置查页表，将当前 K/V 写入物理页。
2. `paged_attention_kernel`：一个 CUDA Block 负责一个 `(request, head)`。

第二个 Kernel 使用 128 个线程，执行四个阶段：

```text
每个线程计算若干 Q·K 分数
          ↓
共享内存归约得到最大值
          ↓
exp(score - max)，再归约得到分母
          ↓
每个线程负责若干 Head Dimension，聚合 softmax(score)·V
```

减去最大值后再计算指数，可避免 Softmax 溢出。分数保存在动态共享内存，归约暂存放在
固定共享内存。Q、K、V、页表和上下文长度在调用期间全部位于 GPU。

## 为什么同步点必须认真处理

共享数组 `reduction[]` 先保存局部最大值，随后又被复用来保存局部指数和。每个线程
读取最终最大值后，必须执行一次 `__syncthreads()`，才能允许其他线程覆盖该数组。
缺少这个屏障时，普通正确性测试可能恰好通过，但 `compute-sanitizer --tool racecheck`
会报告读写冲突。

这个问题说明 CUDA 正确性不能只看数值输出。同步错误具有时序依赖，至少要同时做：

- 独立参考实现的数值对齐；
- `memcheck` 检查越界和非法显存访问；
- `racecheck` 检查共享内存数据竞争。

## 正确性验证

测试配置为 Batch 3、2 Layers、4 Heads、Head Size 64，并故意使用乱序物理页。三组
Context Length 为 `{1,15,16}`、`{17,31,32}`、`{33,64,7}`，因此同时覆盖：

- 长度小于一页、正好一页和刚跨页；
- 31/32/33 的第二个页边界；
- 一个 Batch 内不同上下文长度；
- 新 K/V 写入以及历史 K/V 读取；
- 非连续、乱序的 Block Table。

独立 CPU Reference 使用 double 计算稠密 Attention，GPU 输出结果为：

```text
max_abs_error=4.47035e-08
max_rel_error=0.000356036
max_kv_write_error=0
memcheck: 0 errors
racecheck: 0 hazards
```

运行方法：

```bash
make GPU_COMPUTE_CAPABILITY=86 test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 ./test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck \
  ./test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck \
  ./test_cuda_paged_attention
```

## 性能测试

测试设备为 NVIDIA GeForce RTX 3090，Compute Capability 8.6，编译目标 `sm_86`，
CUDA Runtime 12.5。
固定 12 Heads、Head Size 64、Page Size 16。每个配置预热 20 次；每组连续执行 50 次，
重复 20 组，使用 CUDA Event 计量两个 Kernel 的 GPU 时间。

| Batch | Context | P50 | P95 | 算法有效带宽 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 9.196 us | 9.217 us | 12.027 GB/s |
| 1 | 64 | 13.537 us | 13.558 us | 29.955 GB/s |
| 1 | 256 | 32.768 us | 32.788 us | 48.375 GB/s |
| 1 | 512 | 60.119 us | 60.151 us | 52.529 GB/s |
| 8 | 16 | 9.564 us | 9.586 us | 92.505 GB/s |
| 8 | 64 | 15.503 us | 15.525 us | 209.247 GB/s |
| 8 | 256 | 48.701 us | 48.908 us | 260.387 GB/s |
| 8 | 512 | 90.624 us | 90.686 us | 278.780 GB/s |
| 32 | 16 | 9.994 us | 10.015 us | 354.098 GB/s |
| 32 | 64 | 26.604 us | 26.708 us | 487.760 GB/s |
| 32 | 256 | 82.125 us | 82.209 us | 617.656 GB/s |
| 32 | 512 | 190.945 us | 191.995 us | 529.243 GB/s |

“算法有效带宽”按本算法必须读取或写入的 Q/K/V、输出和新 K/V 字节数除以时间计算，
用于比较同一实现的不同形状。它不是 Nsight Compute 硬件计数器测得的实际 DRAM
流量，缓存命中和重复读取都会使两者不同。

原始结果位于：

- `benchmark/results/cuda_paged_attention_rtx3090.json`
- `benchmark/results/cuda_paged_attention_rtx3090.csv`

复现命令：

```bash
make GPU_COMPUTE_CAPABILITY=86 benchmark_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 ./benchmark_cuda_paged_attention \
  --json benchmark/results/cuda_paged_attention_rtx3090.json \
  --csv benchmark/results/cuda_paged_attention_rtx3090.csv
```

## 怎样理解结果

Batch 1、短 Context 时只有 12 个 CUDA Blocks，GPU 并行度不足，启动开销占比很高。
Batch 增大后，同时执行的 `(request, head)` 增多，有效带宽显著提高。Context 从 256
增加到 512 后，B=32 的有效带宽反而下降，说明当前实现的标量内层循环、重复加载 Q、
共享内存分数数组和线程分工仍有优化空间。

本版本的主要限制是：

- 只支持 FP32；
- 每个 `(request, head)` 固定使用一个 Block；
- Q 在每个 Token 的 dot-product 中重复读取；
- 没有使用 `float4`、Warp Shuffle、Tensor Core 或在线 Softmax 融合；
- 只完成独立 Kernel，尚未形成 GPU 端到端模型推理。

这些限制不是需要隐藏的缺点，而是下一轮优化所需的基线。

## 下一步开发任务

下一阶段先做系统接入，再做算子优化：

1. 为 `GPT2ModelRunner` 增加 GPU KV Cache 所有权和生命周期。
2. 将 Block Table、Context Length、Slot Mapping 持久化到设备侧，按调度增量更新。
3. 接通每层 Q/K/V 输出和本 Kernel，构建 GPU Decode 执行路径。
4. 与 CPU 路径比较完整 logits 和生成 Token，并建立端到端 TTFT/TPOT Benchmark。
5. 在正确链路上增加 FP16、向量化加载、Warp Reduction 和融合版本，逐项比较收益。

面试时可以先讲清楚三个问题：页表怎样把逻辑 Token 映射到物理地址；一个
`(request, head)` 内怎样完成稳定 Softmax；为什么 Racecheck 能发现数值测试没有暴露的
共享内存同步错误。能把这三点讲透，比只背一个性能数字更有说服力。
