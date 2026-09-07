# 开发任务 04：将 CUDA PagedAttention 接入 GPU ModelRunner

**状态：已完成 FP32 端到端基线。**

## 任务目标

任务 03 只证明了独立 PagedAttention Kernel 正确。任务 04 要回答更接近 vLLM 的问题：
Scheduler 产生的请求元数据怎样进入 GPU；BlockManager 的物理页怎样对应设备 KV Cache；
Attention 前后的 Transformer 层在哪里执行；生成 Token 怎样提交回请求状态。

本任务新增约 1600 行 C++/CUDA，其中 GPU ModelRunner 约 760 行，剩余代码主要是共享
元数据、Engine 封装、模型级测试和服务 Benchmark。

## 端到端架构

```text
Sequence / BlockManager / Scheduler（CPU 控制面）
                    │
                    │ ModelInput
                    │ token / position / context length
                    │ slot mapping / block table
                    ▼
GPT2CudaModelRunner（持久化设备内存）
  ├─ GPT-2 权重
  ├─ K/V Cache [page, layer, head, token, dim]
  ├─ 调度元数据 Buffer
  ├─ 中间激活与 logits
  └─ CUDA Stream + cuBLAS Handle
                    │
                    ▼
Embedding → 12 × Transformer Layer → Final LN → LM Head
                    │
                    ▼
设备 Argmax → 只回传 Token ID → Scheduler::commit
```

CPU 和 CUDA Runner 共用 `model_input.hpp`。这样两条路径对 Token 位置、逻辑页、物理页
和页内 Slot 的解释只有一份，避免两个后端各自复制一套容易漂移的索引代码。

## ModelInput 中五个关键数组

| 数组 | 含义 | GPU 中的消费者 |
| --- | --- | --- |
| `token_ids` | 本微步实际处理的 Token | Embedding |
| `positions` | Token 在各自序列中的绝对位置 | Position Embedding |
| `context_lengths` | 当前 Token 可见的完整上下文长度 | PagedAttention |
| `slot_mapping` | 新 K/V 写入的物理页和页内偏移 | KV Write Kernel |
| `block_tables` | 每个逻辑页对应哪个物理页 | Attention 历史 K/V 读取 |

`slot_mapping` 与 `block_tables` 看似重复，职责不同。前者让写入 Kernel 直接找到一个
新 Token 的目标 Slot；后者让 Attention 遍历整个历史上下文。这个划分与服务框架中
“写入位置”和“可见缓存页”两类元数据一致。

## GPU 内存生命周期

`GPT2CudaModelRunner::Impl` 使用 RAII 管理 CUDA Stream、cuBLAS Handle 和所有设备缓冲。
构造时完成一次权重上传和 KV Cache 分配；每轮执行复用同一组激活、logits 和元数据
Buffer，不在 Token 循环中反复 `cudaMalloc`。

模型级测试配置记录的设备内存为：

```text
GPT-2 权重：497,903,616 bytes
3 个 KV Block：3,538,944 bytes
Batch 2 激活与 logits：494,592 bytes
整次动态请求测试上传的调度元数据：1,312 bytes
```

正式服务 Benchmark 配置 16 个 KV Block 和 Batch 4，对应 KV Cache 18,874,368 bytes、
激活与 logits 989,184 bytes。权重和中间激活不回传 CPU；正常执行只回传每个有效请求
的 Greedy Token ID。`last_logits_for_testing()` 是正确性测试专用调试接口。

## 一步 Transformer 怎样执行

每个单 Token 微批次依次执行：

1. GPU Embedding Kernel 读取 Token Embedding 和绝对 Position Embedding。
2. LayerNorm Kernel 在一个 Block 内完成均值和方差归约。
3. cuBLAS 计算 QKV Projection，Bias Kernel 加偏置。
4. Split Kernel 将 `[B, 3C]` 拆为连续的 Q、K、V。
5. CUDA PagedAttention 根据 Slot Mapping 写新 K/V，根据 Block Table 读取历史。
6. cuBLAS 执行 Attention Projection，随后做 Residual Add。
7. 执行第二个 LayerNorm、FC、GELU、Projection 和 Residual Add。
8. 重复 12 层，再执行 Final LayerNorm 和共享词嵌入矩阵的 LM Head。
9. GPU Argmax 对 50,257 个有效词表 logits 归约，只复制 Token ID 到 CPU。

权重在 checkpoint 中按行主序 `[output, input]` 保存。cuBLAS 使用列主序接口，因此调用
`cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, ...)`，把同一段内存解释成转置后的
`[output, input] × [input, batch]`，输出内存在 C++ 侧仍可视为 `[batch, output]`。

## 正确性测试

`dev/cuda/test_gpt2_cuda_model_runner.cu` 使用真实 GPT-2 124M 权重，覆盖：

- Prompt 长度 17 的 16→17 跨页；
- Batch 大小变化和 Prefill/Decode 混合；
- 请求结束后释放 Block，第三个请求强制复用旧物理页；
- GPU 设备 Argmax 与 CPU 完整前缀 Greedy Argmax；
- 最后一个微批次的全部 50,257 个有效 logits；
- 每轮 H2D 字节数只包含五类调度元数据。

结果：

```text
max_abs_logit_error=0.00025177
all greedy tokens equal CPU full-prefix reference
memcheck: 0 errors
racecheck: 0 hazards
```

运行：

```bash
make GPU_COMPUTE_CAPABILITY=86 test_gpt2_cuda_model_runner
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  ./test_gpt2_cuda_model_runner
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  compute-sanitizer --tool memcheck ./test_gpt2_cuda_model_runner
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  compute-sanitizer --tool racecheck ./test_gpt2_cuda_model_runner
```

## 端到端服务 Benchmark

工作负载与任务 02 的 CPU Benchmark 相同：GPT-2 124M，4 个请求在计时起点同时到达，
Prompt 长度 8/16/24/32，每请求生成 4 Token。Engine 初始化和权重 H2D 不计时；先执行
一次 Warmup，再正式重复 3 次。Benchmark 会独立执行 CPU 完整前缀 Greedy Reference，
任一生成 Token 不一致就直接失败。

| 路径 | 中位总时间 | 输出吞吐 | TTFT P50/P95 | TPOT P50/P95 | 请求延迟 P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU Continuous | 2074.0 ms | 7.714 tok/s | 1200.8 / 1812.5 ms | 276.3 / 276.3 ms | 2029.8 / 2067.5 ms |
| CUDA Continuous | 54.107 ms | 295.708 tok/s | 30.912 / 47.232 ms | 1.397 / 19.200 ms | 52.900 / 53.926 ms |

固定负载下 CUDA 吞吐约为 CPU 的 38.3 倍。这个数字只比较本项目同一 GPT-2 权重、
Token、调度配置和输出长度，不用于声称相对 vLLM 或其他推理引擎的性能。

原始结果：

- `benchmark/results/gpt2_cuda_rtx3090.json`
- `benchmark/results/gpt2_cuda_rtx3090.csv`

复现：

```bash
make GPU_COMPUTE_CAPABILITY=86 benchmark_gpt2_cuda_serving
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  ./benchmark_gpt2_cuda_serving --repeats 3 \
  --json benchmark/results/gpt2_cuda_rtx3090.json \
  --csv benchmark/results/gpt2_cuda_rtx3090.csv
```

## Nsight Systems 分析

Profile 包含一次 Warmup 和一次正式运行，共执行 86 个单 Token 微批次。GPU Kernel
时间分布中，cuBLAS GEMV/GEMM 类 Kernel 合计约 63%，PagedAttention 为 8.7%，
LayerNorm 为 8.1%，Bias Kernel 为 6.3%，Argmax 为 5.4%。两次运行共启动 17,576 个
Kernel，平均每个微批次约 204 次 Launch。

这组数据给出两个直接结论：

1. 当前 Prompt 被拆成 T=1 微步，大量线性层退化为小 Batch GEMV，Prefill 没有利用大矩阵乘。
2. 小算子和 Launch 数量过多，Bias/Residual/LayerNorm Fusion 与 CUDA Graph 有实际依据。

Nsight 原始汇总位于：

- `benchmark/results/gpt2_cuda_nsys_cuda_gpu_kern_sum.csv`
- `benchmark/results/gpt2_cuda_nsys_cuda_api_sum.csv`

复现命令：

```bash
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,cublas --sample=none --cpuctxsw=none \
  --output=/tmp/gpt2_cuda_profile \
  ./benchmark_gpt2_cuda_serving --repeats 1 \
  --json /tmp/gpt2_cuda_profile.json --csv /tmp/gpt2_cuda_profile.csv

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  /tmp/gpt2_cuda_profile.nsys-rep
```

Profiler 会增加运行时间，因此不能把 Profile 中的吞吐当成正式性能数字。

## 当前限制和下一步

当前版本完成了端到端系统接入，但仍是正确性优先的 FP32 基线：

- Prefill 仍拆为 T=1 微步；
- 权重和 KV Cache 只有 FP32；
- Bias、Residual、LayerNorm、GELU 等为独立 Kernel；
- 每轮把完整 Block Table 从 CPU 更新到 GPU；
- 没有 Prefix Cache、抢占、CUDA Graph 和 Tensor Parallel。

下一任务优先实现多 Token Prefill。原因是它既能把线性层从 GEMV 提升到 GEMM，也能显著
减少 Prompt 阶段的 Kernel Launch。完成正确性和 Benchmark 后，再依次做低精度、融合、
CUDA Graph、Prefix Cache 和抢占。

该后续任务已经完成，结果见 `task_05_multi_token_prefill_zh.md`。

面试时应能解释：为什么 Slot Mapping 和 Block Table 都需要；为什么权重上传不应计入
每请求 Decode 时间；为什么只有 Token ID 可以回 CPU；以及 Nsight 数据为什么指向
Multi-Token Prefill 和 Fusion，而不是继续只优化已经占 8.7% 的 PagedAttention。
