# Mini-vLLM C++

这是一个基于 [llm.c](https://github.com/karpathy/llm.c) GPT-2 实现构建的教学型
LLM 推理引擎。项目使用 C++ 实现推理执行路径，并参考 vLLM 的核心抽象逐步加入
增量解码、分页 KV Cache、请求调度和连续批处理。

当前版本是经过模型级正确性验证的 CPU 原型。它适合用于学习 LLM 推理引擎中
“请求怎样被调度、KV Cache 怎样分页、模型怎样执行异长 Batch”这条完整主线。

## 当前能力

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| GPT-2 增量解码 | 已完成 | 每步只计算新 Token，复用历史 K/V |
| 分页 KV Cache | 已完成 CPU 版 | 使用 Block Pool 和 Block Table 管理非连续物理页 |
| Sequence | 已完成 | 管理请求状态、Prompt、输出 Token 和已计算 Token 数 |
| BlockManager | 已完成第一版 | 支持 Block 分配、释放、复用和异常检查 |
| Scheduler | 已完成第一版 | 支持 Token Budget、Chunked Prefill 和动态准入/退出 |
| 异长动态 Batch | 已完成 | 每个请求拥有独立 context length |
| Scheduler/ModelRunner 闭环 | 已完成 CPU 基线 | 支持混合 Decode 与 Chunked Prefill |
| 可复现 Benchmark | 已完成 CPU 基线 | 三种模式、warmup、3 次重复、逐请求指标和原始结果 |
| CUDA PagedAttention | 未开始 | CPU 实现将作为正确性参考 |
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
利用较大的矩阵乘，吞吐与连续批处理接近；这说明下一阶段需要多 Token Prefill 和
CUDA kernel，而不能把减少计算量直接等同于端到端加速。

复现方法和指标解释见
[开发任务 02：可复现 Benchmark](doc/task_02_benchmark_zh.md)。

## 下一步开发任务

当前最高优先级任务是实现 CUDA PagedAttention：

1. 保持 Q、分页 K/V、Block Table 和 context length 驻留在 GPU。
2. 使用一个或多个 CUDA Block 处理一个请求的一个 Attention Head。
3. 实现 dot-product reduction、online/稳定 Softmax 和 Value 聚合。
4. 支持异长请求、非连续物理页以及 16/17、32/33 跨页边界。
5. 与现有 CPU PagedAttention reference 对齐，再建立 kernel latency/bandwidth Benchmark。

## 学习文档

- [从 llm.c 到 Mini-vLLM：路线图](doc/mini_vllm_roadmap_zh.md)
- [分页推理原理与实现讲解](doc/paged_inference_learning_zh.md)
- [开发任务 01：接通 Scheduler 与 GPT2ModelRunner](doc/task_01_gpt2_model_runner_zh.md)
- [开发任务 02：可复现推理 Benchmark](doc/task_02_benchmark_zh.md)
- [简历项目表述](doc/resume_project.tex)

## 来源与许可证

本项目基于 Andrej Karpathy 的 [llm.c](https://github.com/karpathy/llm.c)，保留其
Git 历史和 MIT License。Mini-vLLM 模块参考
[vLLM](https://github.com/vllm-project/vllm) 与
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的模块边界独立实现，
没有复制 nano-vLLM 源代码。
