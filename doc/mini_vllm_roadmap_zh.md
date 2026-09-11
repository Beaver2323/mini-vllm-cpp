# 从 llm.c 到 Mini-vLLM

## 项目定位

本项目保留 llm.c 的可读模型与算子实现，参考 nano-vLLM/vLLM 的模块边界，
逐步实现 C++/CUDA 推理引擎。`nano-vllm-reference` 是保留 Git 历史和许可证的参考仓库；
本目录中的 C++ 代码按接口与不变量重新设计，不逐行翻译参考实现。

## 当前完成状态

| 模块 | 状态 | 验证 |
| --- | --- | --- |
| GPT-2 单 Token 增量前向 | 已完成异长动态 Batch 基线 | 活跃批次 1→2→1，逐位置全词表 logits 完全一致 |
| CPU Paged Attention | 已存在 | dense reference 最大绝对误差 1.45372e-07 |
| Sequence | 已完成第一版 | 构造、状态及 Token 记账由调度测试覆盖 |
| BlockManager | 已完成第一版 | 跨块分配、释放、FIFO 复用、double-free 检测 |
| Scheduler | 已完成第一版 | Token Budget、Chunked Prefill、动态加入/退出 |
| 动态 Batch 执行原语 | 已完成 | 每个请求独立 context length，支持中途加入和提前退出 |
| Scheduler/ModelRunner 闭环 | 已完成 CPU/CUDA 基线 | 混合 Decode/Chunked Prefill、设备 Argmax、commit/release |
| 可复现 Benchmark | 已完成 CPU/CUDA 基线 | TTFT/TPOT、吞吐、CSV/JSON 与 Nsight Systems 结果 |
| Prefix Cache | 已完成第一版 | 完整 Block 内容寻址、引用计数共享、LRU 驱逐与 GPU KV 复用 |
| 采样行裁剪 | 已完成 | 零/单/多采样行、Gather、二维 Graph Key、全行参考 |
| Prefix Cache 性能评估 | 已完成 | 16/64/128/256 Token 前缀 off/miss/hit 三组测试 |
| 双 GPU PD 分离 | 已完成功能版 | 主机中转、页映射、首 Token 交接、背压、CPU/单卡与跨卡 logits |
| 抢占 | 未开始 | 后续按学习需要单独设计 |
| CUDA Paged Attention | 已完成并接入 GPT-2 | GPU KV Cache、Slot Mapping、模型级 reference 与 Sanitizer |
| Multi-Token Prefill | 已完成第一版 | Packed Token GEMM、因果分页 Attention、Token Budget Sweep |
| FP16/BF16 | 已完成 | FP16 正式路径、BF16 实验路径、Tensor Core 与 FP32 归约 |
| Residual + LayerNorm / CUDA Graph | 已完成实验版 | 四组 A/B、Graph Replay 与 Nsight Kernel/API 证据 |

模型级测试使用两个独立 GPT-2 实例：reference 对两个请求执行完整前缀前向，incremental
逐 Token 写入分页 KV Cache。请求 0 执行长度 1--33；请求 1 在全局第 5 步加入，执行到
长度 20 后退出，活跃批次随之经历 1→2→1。物理 Block 采用反向、交错分配，覆盖长度
15/16/17 和 31/32/33 页边界；每个有效位置比较 50,257 个词表 logits，最大绝对和相对
误差均为 0。
测试入口为 `dev/test_gpt2_paged_inference.cpp`。

Scheduler、GPT2ModelRunner 和 GPT-2 异长动态 Batch 已经形成端到端闭环。模型级测试
覆盖混合 Decode/Chunked Prefill、动态加入/退出、跨页扩容、Block 回收复用，并确认
greedy 输出与完整前缀前向一致。CPU 与 CUDA Benchmark 均已完成；CUDA PagedAttention
已接入完整 12 层 GPT-2 Decode，模型权重、KV Cache、中间激活和 logits 驻留 GPU。

## 代码地图

```text
mini_vllm/sequence.hpp
  Request 的 token、prompt/completion 数量、computed token 与状态

mini_vllm/block_manager.hpp
  物理 Block 空闲队列、逻辑 Block Table、分配/释放/复用与不变量检查

mini_vllm/scheduler.hpp
  waiting/running 队列、每轮 Token Budget、Chunked Prefill、动态准入/退出

mini_vllm/demo.cpp
  使用确定性假 ModelRunner 展示每轮请求与 Block 状态

mini_vllm/gpt2_model_runner.hpp
  将调度结果拆成动态微批次并构造执行元数据

mini_vllm/gpt2_engine.hpp
  schedule → run → sample → commit/release 闭环

mini_vllm/gpt2_engine_demo.cpp
  使用真实 GPT-2 权重演示连续批处理

dev/test_mini_vllm_control_plane.cpp
  控制面回归测试

paged_kv_cache.hpp
  分页 K/V 存储、PageTable 及每请求独立 context length 的 Attention

dev/test_gpt2_paged_inference.cpp
  两请求动态 Batch 与完整前缀 GPT-2 的全词表正确性对照

dev/test_gpt2_engine.cpp
  Continuous Batching 端到端及完整前缀对齐测试

mini_vllm/cuda/paged_attention.cu
  当前 K/V 写入、分页寻址、稳定 Softmax 与 Value 聚合

dev/cuda/test_paged_attention.cu
  异长请求、乱序物理页与跨页边界的独立稠密参考测试

benchmark/benchmark_cuda_paged_attention.cu
  CUDA Event 计时、P50/P95 与算法有效带宽记录

mini_vllm/cuda/gpt2_cuda_model_runner.cu
  设备内存生命周期、cuBLAS 线性层、Transformer Decode 与 GPU Argmax

mini_vllm/cuda/gpt2_cuda_engine.hpp
  CUDA schedule → run → sample → commit/release 闭环

dev/cuda/test_gpt2_cuda_model_runner.cu
  GPU/CPU 完整词表 logits、生成 Token、跨页和 Block 复用测试
```

编译和运行：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda run -p /home/miniconda3/envs/zyf1 make \
  test_minivllm_control_plane test_gpt2_paged_inference test_gpt2_engine \
  mini_vllm_demo mini_vllm_gpt2_demo
conda run -p /home/miniconda3/envs/zyf1 ./test_minivllm_control_plane
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 ./test_gpt2_paged_inference
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 ./test_gpt2_engine
conda run -p /home/miniconda3/envs/zyf1 ./mini_vllm_demo
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 ./mini_vllm_gpt2_demo
make GPU_COMPUTE_CAPABILITY=86 \
  test_cuda_paged_attention test_gpt2_cuda_model_runner \
  benchmark_cuda_paged_attention benchmark_gpt2_cuda_serving
CUDA_VISIBLE_DEVICES=0 ./test_cuda_paged_attention
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_model_runner
```

## nano-vLLM 参考环境

```text
environment: /home/users/zyf/zyf_llm.c/.conda/nano-vllm
reference:   /home/users/zyf/zyf_llm.c/nano-vllm-reference
model:       /home/users/zyf/zyf_llm.c/models/Qwen3-0.6B
GPU:         2 x RTX 3090 24GB, driver 555.42.06
Python:      3.10
PyTorch:     2.4.1+cu121
Triton:      3.0.0
Transformers: 4.57.6
FlashAttention: 2.7.4.post1
```

Smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 conda run \
  -p /home/users/zyf/zyf_llm.c/.conda/nano-vllm \
  python /home/users/zyf/zyf_llm.c/experiments/nanovllm_smoke.py \
  --model /home/users/zyf/zyf_llm.c/models/Qwen3-0.6B --max-tokens 16
```

两请求端到端生成成功。一次 eager smoke 结果为模型加载和 warmup 64.505 秒、
首次生成 32 Token 用时 45.307 秒。分析源码后发现构造函数只预热 Prefill，
因此实验脚本已增加独立 Decode warmup，并将它排除在正式计时之外。

相同的 2 请求、每请求 16 输出 Token 的 smoke 对比如下：

| 模式 | 模型加载/引擎初始化 | 预热后生成时间 | 输出吞吐 |
| --- | ---: | ---: | ---: |
| eager | 70.755 s | 2.043 s | 15.663 tok/s |
| CUDA Graph | 130.160 s | 1.287 s | 24.871 tok/s |

这个微型负载中 CUDA Graph 生成吞吐约为 eager 的 1.59 倍，同时增加约 59.4 秒初始化成本。
结果证明实验链路和 CUDA Graph 分支可运行，但请求量太小、只运行一次，不能作为正式性能
结论或简历数字。正式 benchmark 需要固定随机工作负载，至少预热一次并重复多轮，报告
TTFT、TPOT、吞吐、P50/P95 延迟和显存占用。

CUDA Graph 模式在 smoke 命令末尾增加 `--cuda-graph`。

## 已完成的 ModelRunner 接口

已实现的 `GPT2ModelRunner` 使用以下核心元数据：

```cpp
struct ModelInput {
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> context_lengths;
    std::vector<int> slot_mapping;
    std::vector<int> block_tables;
    std::vector<int> query_start_locations;
};

class GPT2ModelRunner {
public:
    std::vector<int> run(const SchedulerOutput& scheduled);
};
```

调度元数据的含义：

- `token_ids`：本轮真正计算的 Token，Prefill 可以多个，Decode 通常每请求一个。
- `positions`：每个 Token 在各自序列中的绝对位置。
- `context_lengths`：每个请求当前可见的上下文长度。
- `slot_mapping`：新 K/V 应写入的物理 Block 和页内偏移。
- `block_tables`：Attention 读取历史 K/V 时的逻辑到物理映射。

Chunked Prefill 已将同一调度步中的异长请求压成统一 Token Batch；
`query_start_locations` 标记各请求在 Packed Token 中的起止位置。每个 Token 仍拥有独立
Position、Context Length、Slot Mapping 和 Block Table，因此可以与 Decode 混合执行。

完整前缀重算、分页增量和 Continuous Batching 三组 CPU Benchmark 已完成。在固定
4 请求工作负载中，Continuous Batching 吞吐为分页单请求的 1.99 倍；完整前缀重算
利用多 Token GEMM，吞吐与 Continuous Batching 接近。

独立 CUDA PagedAttention 基线采用一个 CUDA Block 处理一个请求的一个 Attention
Head，Q、分页 K/V、Block Table 和 Context Length 均驻留在 GPU。测试覆盖长度
1/7/15/16/17/31/32/33/64，最大绝对误差为 4.47035e-08；Compute Sanitizer
memcheck 为 0 errors，racecheck 为 0 hazards。在 RTX 3090 的 12 Heads、Head Size 64
配置上，B=32、Context=256 的 Kernel 延迟 P50/P95 为 82.125/82.209 us，算法有效
带宽为 617.656 GB/s。该数字仅代表“新 K/V 写入 + Attention”两个独立 Kernel。

GPU ModelRunner 已完成设备侧端到端 Decode。在与 CPU Benchmark 相同的 4 请求负载
中，RTX 3090 FP32 输出吞吐中位数为 295.708 tok/s，CPU Continuous Batching 为
7.714 tok/s；完整词表 logits 最大绝对误差为 2.5177e-04，所有生成 Token 一致。
Nsight Systems 显示逐 Token 基线中 cuBLAS GEMV/GEMM 类 Kernel 占 GPU Kernel 时间约
63%，PagedAttention 占 8.7%，LayerNorm 占 8.1%。Packed Multi-Token Prefill 将
Token Budget 64 的吞吐提升至 1838.148 tok/s，相对逐 Token GPU 基线提升 6.2 倍，
Profile 中 Kernel Launch 从 17,576 降至 2,176。FP16 权重、激活和分页 KV Cache 已
贯通，LayerNorm、QK/Softmax 和 Value 归约保留 FP32 累加，PagedAttention 和小算子增加
`half2` 向量化。在 RTX 3090、Token Budget 64 上达到 2738.953 tok/s，相对同提交 FP32
的 1866.201 tok/s 提升 46.8%；权重和 KV Cache 显存下降 50%。BF16 路径已实现，但当前
GPT-2 测试出现 Argmax 分歧，仅作为实验模式。

Residual + LayerNorm 融合与 CUDA Graph 已完成。融合 Eager 单独运行是负优化，因此默认
关闭；未融合 Graph 相对未融合 Eager 吞吐提升 15.9%，显式启用 Fusion + Graph 后达到
3189.767 tok/s，相对提升 20.5%。Graph Cache 按 Packed Token 数分桶，动态调度元数据在
Replay 前更新。

完整 Block Prefix Cache 也已接入 Scheduler 和 CPU/CUDA Engine。Cache 使用完整历史前缀
作为内容 Key，通过引用计数共享物理 KV Block，并在内存压力下 LRU 驱逐只有 Cache 引用
的页。GPU 测试中第二个 18 Token 请求命中 16 Token 前缀，只执行剩余 2 Token，生成结果
与 CPU 完整前缀一致。至此 GPT-2 教学型服务主链路已经闭环，后续优先考虑现代模型架构
适配，而不是继续同时增加多个优化功能。

## 学习 nano-vLLM 的顺序

依次阅读：

1. `engine/sequence.py`：请求有哪些状态和计数。
2. `engine/block_manager.py`：Block 所有权和 Prefix Cache 引用计数。
3. `engine/scheduler.py`：每轮选择谁、计算多少 Token。
4. `engine/llm_engine.py`：schedule、run、postprocess 如何闭环。
5. `engine/model_runner.py`：调度结果如何变成设备侧元数据。
6. `layers/attention.py`：Prefill 与 Decode 如何消费 KV Cache。

每读完一个文件，回答三个问题：它拥有什么状态、保持什么不变量、向下一层输出什么。
不要先钻进 Tensor Parallel 或 CUDA Graph；它们建立在上述主链路之上。

## 任务 09—11 完成后的学习安排

必做的采样行裁剪、Prefix Cache 性能对照以及双 GPU PD 功能均已完成，见
[结果和复现](../benchmark/results/task09_11/README.md)。先按
[任务 09](task_09_sample_rows_zh.md) → [任务 10](task_10_prefix_benchmark_zh.md) →
[任务 11](task_11_pd_disaggregation_zh.md) 跟踪代码和运行测试，再决定后续方向。

当前没有继续扩展功能的必做项。PD 的 pinned buffer 复用、异步传输、两卡完整请求副本对照、
TP、抢占和 HTTP 服务都留作可选工作，本轮未扩展。PD 当前偏功能学习，保留真实的负性能结果。
