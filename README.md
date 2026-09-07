# Mini-vLLM C++

这是一个基于 [llm.c](https://github.com/karpathy/llm.c) GPT-2 实现构建的教学型
LLM 推理引擎。项目使用 C++ 实现推理执行路径，并参考 vLLM 的核心抽象逐步加入
增量解码、分页 KV Cache、请求调度和连续批处理。

当前版本同时包含 CPU 正确性基线和端到端 FP32 CUDA Decode 路径。Scheduler 产生的
Token、Position、Context Length、Slot Mapping 与 Block Table 会进入 GPU ModelRunner；
模型权重、分页 KV Cache、中间激活和 logits 在设备侧持久保存，只把最终 Greedy Token
传回 CPU。GPU Prefill 会把同一调度轮的 Token 压紧为 `total_tokens` Batch，使用 GEMM
完成各层投影。

## 当前能力

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| GPT-2 增量解码 | 已完成 | 每步只计算新 Token，复用历史 K/V |
| 分页 KV Cache | 已完成 CPU 版 | 使用 Block Pool 和 Block Table 管理非连续物理页 |
| Sequence | 已完成 | 管理请求状态、Prompt、输出 Token 和已计算 Token 数 |
| BlockManager | 已完成第一版 | 支持 Block 分配、释放、复用和异常检查 |
| Scheduler | 已完成第一版 | 支持 Token Budget、Chunked Prefill 和动态准入/退出 |
| 异长动态 Batch | 已完成 | 每个请求拥有独立 context length |
| Scheduler/ModelRunner 闭环 | 已完成 CPU/CUDA 基线 | 支持混合 Decode、Chunked Prefill、动态请求和页复用 |
| 可复现 Benchmark | 已完成 CPU/CUDA 基线 | warmup、重复测试、TTFT/TPOT、吞吐和原始结果 |
| CUDA PagedAttention | 已完成并接入模型 | FP32 Decode，设备侧页表、Slot Mapping 与 KV Cache |
| Multi-Token Prefill | 已完成第一版 | Packed Token Batch、因果分页 Attention、混合 Prefill/Decode |
| Prefix Cache 与抢占 | 未开始 | Block 引用计数接口已经预留 |

## 架构

```text
请求
  │
  ▼
Sequence：保存 Token、状态和计算进度
  │
  ▼
Scheduler：在序列数与 Token Budget 内选择本轮工作
  │
  ▼
BlockManager：为本轮 Token 保证足够的 KV Block
  │
  ▼
GPT2ModelRunner：整理输入 Token、位置、context length 和 Block Table
  │
  ▼
GPT-2 增量前向与 PagedAttention
  │
  ▼
Sampler：生成新 Token
  │
  ▼
Scheduler::commit：更新状态，完成时释放 Block
```

图中的模块已经连接成端到端执行循环。Scheduler 分配的物理 Block ID 会直接映射到
KVCachePool；ModelRunner 将 Chunked Prefill 拆成动态微批次，并在完成输入后执行
greedy sampling 和状态提交。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `mini_vllm/sequence.hpp` | 请求状态和 Token 生命周期 |
| `mini_vllm/block_manager.hpp` | KV Block 所有权、分配与回收 |
| `mini_vllm/scheduler.hpp` | Token Budget 和请求调度 |
| `mini_vllm/gpt2_model_runner.hpp` | 调度元数据、动态微批次和 GPT-2 执行 |
| `mini_vllm/gpt2_engine.hpp` | schedule、run、sample、commit 执行闭环 |
| `mini_vllm/demo.cpp` | 不依赖模型的调度过程演示 |
| `mini_vllm/gpt2_engine_demo.cpp` | 使用真实 GPT-2 权重的连续批处理演示 |
| `paged_kv_cache.hpp` | 分页 KV Cache 与 CPU PagedAttention |
| `train_gpt2.cpp` | GPT-2 单 Token 和动态 Batch 增量前向 |
| `dev/test_mini_vllm_control_plane.cpp` | 调度和缓存管理测试 |
| `dev/test_paged_attention_resume.cpp` | PagedAttention 稠密参考测试 |
| `dev/test_gpt2_paged_inference.cpp` | GPT-2 模型级全词表正确性测试 |
| `dev/test_gpt2_engine.cpp` | Continuous Batching 端到端测试 |
| `mini_vllm/model_input.hpp` | CPU/CUDA 共用的调度元数据构造与校验 |
| `mini_vllm/cuda/paged_attention.cu` | CUDA KV 写入与 PagedAttention Decode Kernel |
| `mini_vllm/cuda/gpt2_cuda_model_runner.cu` | GPU 权重、KV Cache、Transformer 层与设备 Argmax |
| `mini_vllm/cuda/gpt2_cuda_engine.hpp` | CUDA schedule、run、sample、commit 执行闭环 |
| `dev/cuda/test_paged_attention.cu` | CUDA Kernel 与独立 CPU 稠密参考对照 |
| `dev/cuda/test_gpt2_cuda_model_runner.cu` | GPU ModelRunner 端到端正确性测试 |
| `benchmark/benchmark_cuda_paged_attention.cu` | CUDA Kernel 延迟与有效带宽测试 |
| `benchmark/benchmark_gpt2_cuda_serving.cu` | GPU 服务 TTFT、TPOT 与吞吐测试 |
| `doc/mini_vllm_roadmap_zh.md` | 开发路线、实验结果和学习顺序 |
| `doc/paged_inference_learning_zh.md` | 分页推理原理与代码讲解 |

## 构建

控制面测试和演示不需要模型权重：

```bash
make test_minivllm_control_plane mini_vllm_demo
./test_minivllm_control_plane
./mini_vllm_demo
```

模型级测试需要在仓库根目录放置 GPT-2 124M 权重文件 `gpt2_124M.bin`：

```bash
make test_gpt2_paged_inference test_gpt2_engine mini_vllm_gpt2_demo
OMP_NUM_THREADS=16 ./test_gpt2_paged_inference
OMP_NUM_THREADS=16 ./test_gpt2_engine
OMP_NUM_THREADS=16 ./mini_vllm_gpt2_demo
```

在本项目的开发机器上，可使用既有 Conda 环境：

```bash
conda run -p /home/miniconda3/envs/zyf1 make \
  test_minivllm_control_plane mini_vllm_demo test_gpt2_paged_inference \
  test_gpt2_engine mini_vllm_gpt2_demo

conda run -p /home/miniconda3/envs/zyf1 ./test_minivllm_control_plane
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 \
  ./test_gpt2_paged_inference
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 \
  ./test_gpt2_engine
```

CUDA PagedAttention 可独立构建和验证；GPU ModelRunner 测试需要 GPT-2 权重：

```bash
make GPU_COMPUTE_CAPABILITY=86 \
  test_cuda_paged_attention test_gpt2_cuda_model_runner \
  benchmark_cuda_paged_attention benchmark_gpt2_cuda_serving
CUDA_VISIBLE_DEVICES=0 ./test_cuda_paged_attention
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  ./test_gpt2_cuda_model_runner
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck \
  ./test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck \
  ./test_cuda_paged_attention
```

## 正确性验证

模型级回归包含两个异长请求：

- 请求 0 执行长度 1 到 33。
- 请求 1 在全局第 5 步加入，执行到长度 20 后退出。
- 活跃 Batch 大小经历 `1 → 2 → 1`。
- 物理页使用反向、交错映射。
- 覆盖 15/16/17 和 31/32/33 分页边界。
- 每个有效位置比较全部 50,257 个词表 logits。

当前结果：

```text
PagedAttention dense reference max_abs_error=1.45372e-07
GPT-2 incremental inference max_abs_error=0
GPT-2 incremental inference max_rel_error=0
GPT2Engine full-prefix greedy agreement=passed
CUDA PagedAttention max_abs_error=4.47035e-08
CUDA PagedAttention KV write max_error=0
Compute Sanitizer memcheck=0 errors, racecheck=0 hazards
CUDA GPT2ModelRunner max_abs_logit_error=0.00025177
CUDA GPT2ModelRunner CPU greedy agreement=passed
```

## CPU Benchmark

固定 4 个请求，Prompt 长度为 8/16/24/32，每个请求输出 4 Token。每种模式先 warmup
一次，再正式运行 3 次；下表为跨重复的中位数：

| 模式 | 总时间 | 输出吞吐 | TTFT P50/P95 | TPOT P50/P95 |
| --- | ---: | ---: | ---: | ---: |
| 完整前缀重算 | 2.212 s | 7.234 tok/s | 733.0 / 1507.9 ms | 134.4 / 195.4 ms |
| 分页单请求 | 4.127 s | 3.877 tok/s | 1822.6 / 3761.7 ms | 45.0 / 45.2 ms |
| 连续批处理 | 2.074 s | 7.714 tok/s | 1200.8 / 1812.5 ms | 276.3 / 276.3 ms |

Continuous Batching 在该工作负载中的吞吐是分页单请求的 1.99 倍。CPU 上完整前缀重算
利用较大的矩阵乘，吞吐与连续批处理接近；这说明 Prefill 需要保留多 Token GEMM，
不能把减少计算量直接等同于端到端加速。

复现方法和指标解释见
[开发任务 02：可复现 Benchmark](doc/task_02_benchmark_zh.md)。

## CUDA PagedAttention Benchmark

RTX 3090、`sm_86`、12 Heads、Head Size 64、Page Size 16，计时范围包含“当前 Token K/V
写入 + PagedAttention”两个 Kernel。每个配置预热 20 次；每组连续执行 50 次，
重复 20 组后报告组均值的 P50/P95：

| Batch | Context | 延迟 P50/P95 | 有效带宽 |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 9.196 / 9.217 us | 12.027 GB/s |
| 1 | 512 | 60.119 / 60.151 us | 52.529 GB/s |
| 8 | 256 | 48.701 / 48.908 us | 260.387 GB/s |
| 32 | 256 | 82.125 / 82.209 us | 617.656 GB/s |
| 32 | 512 | 190.945 / 191.995 us | 529.243 GB/s |

这里的有效带宽按算法所需的 Q/K/V、输出和新 K/V 字节数计算，不等于硬件计数器测得的
DRAM 带宽。该结果衡量独立 FP32 Kernel，不能代表完整模型的端到端吞吐。
完整 12 组原始结果见 `benchmark/results/cuda_paged_attention_rtx3090.json` 和 `.csv`，
实现与分析见 [开发任务 03：CUDA PagedAttention](doc/task_03_cuda_paged_attention_zh.md)。

## GPU 端到端 Benchmark

与 CPU Benchmark 使用同一 GPT-2 124M 权重、Prompt Token 和输出长度。RTX 3090、
FP32、4 个请求同时到达，每个请求输出 4 Token；预热一次后正式重复 3 次：

| 路径 | 中位总时间 | 输出吞吐 | TTFT P50/P95 | TPOT P50/P95 |
| --- | ---: | ---: | ---: | ---: |
| CPU Continuous Batching | 2074.0 ms | 7.714 tok/s | 1200.8 / 1812.5 ms | 276.3 / 276.3 ms |
| CUDA 逐 Token Prefill | 54.107 ms | 295.708 tok/s | 30.912 / 47.232 ms | 1.397 / 19.200 ms |
| CUDA Packed Prefill | 8.704 ms | 1838.148 tok/s | 2.511 / 4.042 ms | 1.541 / 1.789 ms |

Packed Prefill 相对逐 Token CUDA 基线吞吐提升 6.2 倍。16 个生成 Token 与独立 CPU
完整前缀 Greedy Reference 全部一致。该数字用于本项目版本间回归，不代表 vLLM、
其他模型、精度或工作负载的通用加速比。

Nsight Systems 显示 GPU Kernel 时间主要由 cuBLAS GEMV/GEMM 类 Kernel 占用约 63%，
PagedAttention 占 8.7%，LayerNorm 占 8.1%；两次被分析运行共启动 17,576 个 Kernel，
Packed Prefill 将相同 Profile 的 Launch 数降至 2,176，减少 87.6%。

完整说明见 [开发任务 04：GPU ModelRunner](doc/task_04_gpu_model_runner_zh.md)。
Packed Prefill 设计与 Token Budget 曲线见
[开发任务 05：Multi-Token Prefill](doc/task_05_multi_token_prefill_zh.md)。

## 下一步开发任务

当前最高优先级任务是增加低精度执行：

1. 增加 FP16/BF16 权重和 KV Cache，使用 Tensor Core。
2. 分别校验 logits、生成 Token、显存占用和吞吐变化。
3. 融合 Bias、Residual、LayerNorm 等小 Kernel。
4. 为固定 Batch Bucket 捕获 CUDA Graph，并与 Eager 路径对照。
5. 在控制面实现 Prefix Cache、引用计数与抢占。

## 学习文档

- [从 llm.c 到 Mini-vLLM：路线图](doc/mini_vllm_roadmap_zh.md)
- [分页推理原理与实现讲解](doc/paged_inference_learning_zh.md)
- [开发任务 01：接通 Scheduler 与 GPT2ModelRunner](doc/task_01_gpt2_model_runner_zh.md)
- [开发任务 02：可复现推理 Benchmark](doc/task_02_benchmark_zh.md)
- [开发任务 03：CUDA PagedAttention](doc/task_03_cuda_paged_attention_zh.md)
- [开发任务 04：GPU ModelRunner](doc/task_04_gpu_model_runner_zh.md)
- [开发任务 05：Multi-Token Prefill](doc/task_05_multi_token_prefill_zh.md)
- [简历项目表述](doc/resume_project.tex)

## 来源与许可证

本项目基于 Andrej Karpathy 的 [llm.c](https://github.com/karpathy/llm.c)，保留其
Git 历史和 MIT License。Mini-vLLM 模块参考
[vLLM](https://github.com/vllm-project/vllm) 与
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的模块边界独立实现，
没有复制 nano-vLLM 源代码。
