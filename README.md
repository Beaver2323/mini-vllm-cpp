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
| 异长动态 Batch | 已完成执行原语 | 每个请求拥有独立 context length |
| Scheduler/ModelRunner 闭环 | 开发中 | 下一阶段核心任务 |
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

目前图中的 Sequence、Scheduler、BlockManager、GPT-2 增量前向和 PagedAttention
已经分别实现。下一步要完成 GPT2ModelRunner，把这些模块连接成同一个端到端循环。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `mini_vllm/sequence.hpp` | 请求状态和 Token 生命周期 |
| `mini_vllm/block_manager.hpp` | KV Block 所有权、分配与回收 |
| `mini_vllm/scheduler.hpp` | Token Budget 和请求调度 |
| `mini_vllm/demo.cpp` | 不依赖模型的调度过程演示 |
| `paged_kv_cache.hpp` | 分页 KV Cache 与 CPU PagedAttention |
| `train_gpt2.cpp` | GPT-2 单 Token 和动态 Batch 增量前向 |
| `dev/test_mini_vllm_control_plane.cpp` | 调度和缓存管理测试 |
| `dev/test_paged_attention_resume.cpp` | PagedAttention 稠密参考测试 |
| `dev/test_gpt2_paged_inference.cpp` | GPT-2 模型级全词表正确性测试 |
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
make test_gpt2_paged_inference
OMP_NUM_THREADS=16 ./test_gpt2_paged_inference
```

在本项目的开发机器上，可使用既有 Conda 环境：

```bash
conda run -p /home/miniconda3/envs/zyf1 make \
  test_minivllm_control_plane mini_vllm_demo test_gpt2_paged_inference

conda run -p /home/miniconda3/envs/zyf1 ./test_minivllm_control_plane
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 \
  ./test_gpt2_paged_inference
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
```

## 下一步开发任务

当前最高优先级任务是实现 `GPT2ModelRunner`，形成端到端 Continuous Batching：

1. 定义 `ModelInput`，统一保存 Token、绝对位置、context length、slot mapping 和 Block Table。
2. 增加独立推理工作区，使增量推理不依赖训练前向分配激活内存。
3. 将 `SchedulerOutput` 转换为模型输入，支持一个调度轮次同时包含 Decode 和 Chunked Prefill。
4. 让 BlockManager 的物理 Block ID 直接对应 KV Cache Pool 中的物理页。
5. 增加 Greedy Sampler，并通过 `Scheduler::commit` 更新请求或释放完成请求的 Block。
6. 新增端到端测试，验证请求动态加入、提前退出、跨页扩容和回收后复用。

完成该任务后，再依次开发正式 Benchmark、CUDA PagedAttention、Prefix Cache/抢占。

## 学习文档

- [从 llm.c 到 Mini-vLLM：路线图](doc/mini_vllm_roadmap_zh.md)
- [分页推理原理与实现讲解](doc/paged_inference_learning_zh.md)
- [开发任务 01：接通 Scheduler 与 GPT2ModelRunner](doc/task_01_gpt2_model_runner_zh.md)
- [简历项目表述](doc/resume_project.tex)

## 来源与许可证

本项目基于 Andrej Karpathy 的 [llm.c](https://github.com/karpathy/llm.c)，保留其
Git 历史和 MIT License。Mini-vLLM 模块参考
[vLLM](https://github.com/vllm-project/vllm) 与
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的模块边界独立实现，
没有复制 nano-vLLM 源代码。
