# 从 PyTorch 开发者到推理引擎开发者：零基础学习入口

需要准备项目面试时，配合 [面试理解与复述路线](../interview/README.md)：先学原理，再用短答、追问与小实验检验。

这条路线专门面向你：会看 PyTorch 的 Module、Tensor、算子、Dispatcher 或编译链路，
但还没有接触过 vLLM。你不需要先装 vLLM，也不需要先读完 CUDA Kernel。

任务 01—11 按开发顺序组织，现已逐篇补齐详细源码精读。这里按理解所需的前置知识重排：
先理解为什么有这些模块，再进入对应任务读关键代码、调用点、手算和练习答案。
全部原任务入口见 [逐任务源码精读索引](../paged_inference_learning_zh.md#逐任务源码精读怎么使用)。

## 1. 先把自己的经验放到正确位置

你熟悉的框架问题可能是：这个算子如何注册，Tensor 怎样 Dispatch 到设备，FX 图如何分解、
Lowering 和生成 Kernel。推理引擎会继续使用这些能力，但还需要持续决定：

- 哪几个请求现在可以执行，各自执行几个 Token？
- 请求前几轮算出的 K/V 存在哪里，谁拥有这块内存？
- 某个请求完成后，怎样让新请求立即利用腾出的容量？
- 本轮输出属于谁，要继续计算还是结束？

先从你熟悉的 `model(input)` 往外扩一层，再往里看缓存，而不是先记一长串框架术语。

## 2. 两条阅读路线

**第一遍，只建立能运行的理解：**

```text
第 1 节：一次 forward → 生成循环 → KV Cache
   ↓
第 2 节：一个请求 → 多请求调度 → 本轮做哪些 Token
   ↓
第 3 节：连续 KV Tensor → 分页池 → Packed 元数据
```

**第二遍，再接上你的框架开发经验：**

```text
第 4 节：PyTorch 算子 → 本项目 CUDA Runner → 编译图/执行图/数据缓存
   ↓
第 5 节：读本地 nano-vLLM 的 Python → 对照项目 → 再进入 vLLM V1
   ↓
第 6 节：实验、断点、术语与自测答案
```

| 节 | 文档 | 这一节结束时你应该能做什么 |
| --- | --- | --- |
| 1 | [生成与 KV Cache](01_generation_and_kv.md) | 解释为什么生成的 Token 下一轮才产生 KV |
| 2 | [请求与调度](02_requests_and_scheduler.md) | 手推 demo 四轮的输入、输出、状态和空闲页 |
| 3 | [分页与 Packed 输入](03_pages_and_packed.md) | 从逻辑 Token 算到物理页，再还原元数据数组 |
| 4 | [从 PyTorch 到 CUDA 执行](04_pytorch_to_cuda.md) | 跟踪一次 step 的 CPU/GPU 边界，区分三种“图/缓存” |
| 5 | [读懂 nano-vLLM，再看 vLLM](05_read_nanovllm_and_vllm.md) | 用 Python 对照理解职责，识别不同版本的实现差异 |
| 6 | [实验与自测手册](06_labs_and_answers.md) | 独立运行、下断点、解释结果并回答常见问题 |

每次只完成一节的过关任务，先写出自己的预测，再运行验证。遇到生词可以查第 6 节，不必
打开几十个网页。读完前三节后，先用任务 01、05 对照状态与元数据，再逐步进入 CUDA 和
任务 09、10、11；不要因为后者是最新开发任务就跳过前置实现。

## 3. 第一小时只做这两件事

先运行一个普通 PyTorch 小模型，再运行项目真实 Scheduler 的控制面 demo：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda activate zyf1
python doc/from_pytorch/examples/attention_and_pages.py cache
c++ -std=c++17 -O0 -g -gdwarf-4 -I. mini_vllm/demo.cpp -o /tmp/zyf_learning_demo
/tmp/zyf_learning_demo
```

第一个实验使用 CPU 上随机初始化的两层小模型；第二个 demo 不执行神经网络，只用确定性
假 Token 驱动真实 Sequence、BlockManager 和 Scheduler。都不需要下载权重或使用 GPU。
它们分别帮你观察“模型计算”和“请求管理”，随后再看两者在 CUDA Engine 中的连接。

本次已在 `zyf1` 的 Python 3.8.20 / PyTorch 2.4.1+cu121 上运行通过，输出保存在
[学习实验记录](examples/verified_output.txt)。模型使用 float64 保持数值对照清楚，不能用该实验
推断真实 GPT-2 的速度。

## 4. 与原任务文档怎么接起来

| 入门知识 | 后续实现手册 | 应优先读的函数 |
| --- | --- | --- |
| 单请求生成、记账 | [任务 01](../task_01_gpt2_model_runner_zh.md) | `Sequence::append_token`、`Scheduler::commit` |
| KV 分页与地址 | [任务 03](../task_03_cuda_paged_attention_zh.md) | `cache_offset`、`write_kv_cache_kernel` |
| Packed 元数据 | [任务 05](../task_05_multi_token_prefill_zh.md) | `prepare_packed_model_input` |
| 设备上的完整模型 | [任务 04](../task_04_gpu_model_runner_zh.md) | `Impl::forward<T>` |
| 存储与累加精度 | [任务 06](../task_06_mixed_precision_zh.md) | `matmul`、`logits_matmul` |
| 融合与重放 | [任务 07](../task_07_fusion_cuda_graph_zh.md) | `fused_residual_layernorm`、`graph_key` |
| 跨请求复用前缀 | [任务 08](../task_08_prefix_cache_zh.md) | `apply_prefix_cache`、`cache_computed_prefix_blocks` |
| 只计算有效输出行 | [任务 09](../task_09_sample_rows_zh.md) | `last_logit_token_indices_`、Gather Kernel |
| 缓存是否节省时间 | [任务 10](../task_10_prefix_benchmark_zh.md) | `measure`、off/miss/hit 循环 |
| 双卡阶段交接 | [任务 11](../task_11_pd_disaggregation_zh.md) | `try_handoff`、`copy_kv_to` |

各任务前面的源码精读说明当前实现，后面的历史性能表保留当时的数据。结合
[任务 09—11 实测](../../benchmark/results/task09_11/README.md)，不要把历史缓冲区大小当成现在的常量。

## 5. 阅读代码时统一使用的记号

| 记号 | 意义 | 最容易混淆的地方 |
| --- | --- | --- |
| B | 本轮请求数 | 与 GPU Packed 输入行数 N 不一定相同 |
| T | 某个请求的完整上下文长度 | 不一定都是本轮要重新计算的 Token |
| N | 本轮实际输入 Token 总数 | 多请求的 scheduled token 数之和 |
| R | 本轮需要采样的行数 | 未完成的 Prefill Chunk 不产生样本 |
| C | hidden size | GPT-2 124M 为 768 |
| H / D | 注意力头数 / 每头维度 | 本项目 MHA 下 C = H × D |
| L | Transformer 层数 | 每层有自己的 K/V |
| P | 每页 Token 数 | 教学 demo 为 4，正式 CUDA 为 16 |
| V | 有效词表大小 | 与用于对齐分配的 padded_vocab_size 不同 |

本项目原代码有些参数仍叫 `batch_size`、`request`：在 Packed CUDA 路径中，它们可能表示
**Token 行数或 Token 行号**。遇到变量名时优先查生产者和实际形状，第 3、4 节会具体示范。

## 6. 这一轮学习先不要求掌握什么

前三节不涉及 TP、PP、EP、RoPE、FlashAttention 内核推导、HTTP 服务或多机通信。
这些会在第 5、6 节说明与已有知识的联系，暂时只需知道它们分别解决什么问题。

目标是能从一个 Prompt 出发，讲清它每一轮在哪里、算了什么、哪些数据被复用、谁决定它
完成。掌握这条链路后，再读 vLLM 的复杂实现会更有方向。
