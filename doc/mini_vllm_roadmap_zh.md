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
| Scheduler/ModelRunner 闭环 | 已完成 CPU 基线 | 混合 Decode/Chunked Prefill、greedy sample、commit/release |
| 抢占与 Prefix Cache | 未完成 | Block 引用计数已预留 |
| CUDA Paged Attention | 未完成 | CPU 实现作为后续 reference |

模型级测试使用两个独立 GPT-2 实例：reference 对两个请求执行完整前缀前向，incremental
逐 Token 写入分页 KV Cache。请求 0 执行长度 1--33；请求 1 在全局第 5 步加入，执行到
长度 20 后退出，活跃批次随之经历 1→2→1。物理 Block 采用反向、交错分配，覆盖长度
15/16/17 和 31/32/33 页边界；每个有效位置比较 50,257 个词表 logits，最大绝对和相对
误差均为 0。
测试入口为 `dev/test_gpt2_paged_inference.cpp`。

Scheduler、GPT2ModelRunner 和 GPT-2 异长动态 Batch 已经形成端到端闭环。模型级测试
覆盖混合 Decode/Chunked Prefill、动态加入/退出、跨页扩容、Block 回收复用，并确认
greedy 输出与完整前缀前向一致。当前仍是 CPU 正确性基线，尚未完成性能 Benchmark。

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

Chunked Prefill 在 ModelRunner 内拆成单 Token 微步，每个微步压紧当前仍有工作的请求。
这种实现先保证异长 Batch 的映射正确；后续可增加多 Token Prefill 专用执行路径。

下一阶段先建立完整前缀重算、分页增量和 Continuous Batching 三组可信 Benchmark，
再将 CPU PagedAttention 替换为 CUDA kernel。

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
