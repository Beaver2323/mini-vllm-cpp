# 第 5 节：读懂 nano-vLLM 的 Python，再进入 vLLM

上一节：[PyTorch 到 CUDA](04_pytorch_to_cuda.md) · [目录](README.md) · 下一节：[实验与答案](06_labs_and_answers.md)

前四节已经建立请求、计算和缓存的联系。现在用本地 Python 参考实现巩固，再看更完整的 vLLM。

## 1. 先记住这三份代码的关系

| 名称 | 在本次学习中的作用 | 本机位置/参考 |
| --- | --- | --- |
| mini-vLLM C++ | 你实际开发、运行和写入简历的项目 | 当前 `llm.c` 仓库 |
| nano-vLLM | 用 PyTorch/Python 阅读相似模块职责的参考 | `../nano-vllm-reference` |
| vLLM | 理解成熟引擎的架构与设计边界 | 官方文档与源码 |

它们不是同一个包，也不能把其中一个文件名和行为直接套在另一个实现上。本章本地源码核对
基于 nano-vLLM 提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，网页链接固定到该提交。
vLLM 官方设计说明核对日期为 2026-09-11；latest 页面之后可能变化。

当前 zyf1 是 Python 3.8.20，而这份 nano-vLLM 的
[pyproject.toml](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/pyproject.toml)
要求 Python >=3.10,<3.13，并依赖 FlashAttention、Triton 等。这里先做源码阅读；前三节的实验
直接用 zyf1 现有 PyTorch 即可运行，阅读 nano 源码不需要 import 整个包。

## 2. 最短阅读顺序：先 forward，再向两端扩展

你是 PyTorch 开发者，可以先从最熟悉的 Module 进去：

```text
models/qwen3.py：Qwen3Attention.forward
    ↓ 看到 self.attn(q,k,v)
layers/attention.py：Attention.forward
    ↓ 发现 get_context() 提供页表、Slot 和长度
engine/model_runner.py：prepare_prefill / prepare_decode
    ↓ 查这些输入由谁决定
engine/scheduler.py：schedule / postprocess
    ↓ 查外部如何循环调用
engine/llm_engine.py：step / generate
```

再从 LLMEngine 正向走一遍，就得到完整链路。这种“两端相接”的读法能避免一开始陷入模型
加载、分布式进程、配置解析等长初始化流程。

## 3. 每个阅读点具体到哪里看

| 顺序 | nano-vLLM 源码（固定版本） | 重点函数 | 本项目对应位置 |
| --- | --- | --- | --- |
| 1 | [models/qwen3.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/models/qwen3.py#L72) | Qwen3Attention.forward | CUDA Runner 的 QKV/Split/Attention/Projection |
| 2 | [layers/attention.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L59) | Attention.forward | paged_attention.cu |
| 3 | [utils/context.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/utils/context.py) | Context / set_context / get_context | ModelInput 和 Runner 的设备元数据 Buffer |
| 4 | [engine/model_runner.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L129) | prepare_prefill / prepare_decode | prepare_packed_model_input |
| 5 | [engine/scheduler.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py) | schedule / postprocess | Scheduler::schedule / commit |
| 6 | [engine/llm_engine.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py#L48) | step / generate | Engine::step / Benchmark 外层 while |
| 7 | [engine/block_manager.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py) | allocate / hash_blocks / deallocate | BlockManager 的分配/注册前缀/释放 |
| 8 | [layers/embed_head.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/embed_head.py#L56) | ParallelLMHead.forward | Gather → logits_matmul |

在本机可直接定位：

```bash
rg -n 'def forward|self.attn' ../nano-vllm-reference/nanovllm/models/qwen3.py
rg -n 'prepare_prefill|prepare_decode|set_context' ../nano-vllm-reference/nanovllm/engine/model_runner.py
rg -n 'def schedule|def postprocess|def preempt' ../nano-vllm-reference/nanovllm/engine/scheduler.py
```

## 4. 第一个熟悉的入口：Qwen3Attention.forward

你会看到正常的 PyTorch 张量运算：QKV Projection、split、view、位置编码、Attention、输出投影。
其中这两句仍然是标准 Module 组合：

```python
o = self.attn(q, k, v)
output = self.o_proj(o.flatten(1, -1))
```

你的第一个追问应是：调用 self.attn 时怎么知道历史 K/V 在哪里？它的显式参数只有 q/k/v。
答案在 `Attention.forward` 读取的 Context。这里 Context 是本轮输入元数据，不是
`autograd.Function` 的 ctx，也不是只代表聊天文本的“上下文”。

注意 nano 这里是 Qwen3，包含 GPT-2 没有的 RoPE、RMSNorm、GQA 等模型结构。比较的是模块
职责和数据流，不要求两种模型相同 Token 得到相同 logits。

## 5. 从 Attention 的消费者找到 Context 的生产者

`Attention.forward` 从 Context 读取 slot_mapping、block_tables、context_lens 或前缀和长度。
它先把新 K/V 写入缓存，再按 Prefill/Decode 分支调用相应 Attention 接口。

向上看 ModelRunner：

- `prepare_prefill` 根据 `num_cached_tokens` 和 `num_scheduled_tokens` 打包本轮输入。
- `prepare_decode` 从每个请求取 last_token，设置位置为 `len(seq)-1`、可见长度为 `len(seq)`。
- `set_context` 把页表等元数据传给模型里的 Attention 消费。

这些对应本项目的显式 ModelInput。尤其是 `cu_seqlens_q`，它是各请求本轮 Q 长度的前缀和，
你已经在 `query_start_locations=[0,1,4]` 中见过同样的边界概念。存在缓存时，Q 长度只含新输入，
K 长度包含历史，二者前缀和可能不同。

阅读时可画这张依赖图：

```text
Scheduler 决定 num_scheduled_tokens
    ↓
ModelRunner 把 Sequence 状态变为设备 Tensor，并设置 Context
    ↓
普通 Module 前向经过 Attention
    ↓
Attention 读取 Context，找准当前写入 Slot 与历史读取页表
```

## 6. 再向上读，看到熟悉的三个步骤

nano-vLLM 的 `LLMEngine.step` 中同样可以找到：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

它与本项目 `schedule → run → commit` 的职责相似。但 `is_prefill` 是这份 nano 代码整个 Batch
的阶段标记；本项目每个 ScheduledItem 单独带 phase，所以能同轮混合 Prefill 与 Decode。
职责相似，不代表输入协议可以直接互换。

## 7. 必须知道的实现差异

下面均来自这份本地固定提交与当前 C++ 代码对照，不能泛化为所有 nano/vLLM 版本。

| 维度 | 本地 nano-vLLM | 本项目 C++ |
| --- | --- | --- |
| 模型计算 | PyTorch Qwen3 Module + Attention 后端 | 手工 GPT-2 CUDA 前向 |
| 调度 | 有待 Prefill 工作时先返回 Prefill Batch；否则 Decode | Running 先推进，再接纳 Waiting；可混合阶段 |
| Chunked Prefill | 支持；代码限制只允许本批第一个请求被切块 | count 由剩余预算和 pending 决定 |
| 抢占 | preempt 释放页，重新排队计算 | 单卡未实现抢占；无进展时报错 |
| 前缀 Key | 前一块 hash 链接当前块 Token；命中还检查当前块 Token | 完整历史 Token 向量作为 map Key |
| 缓存引用 | ref_count 统计正在使用的引用，0 引用页仍可能保有 hash 记录 | Cache 自身额外持有一次引用；cache-only 是 1 |
| 缓存布局 | 总池 `[2,L,pages,P,Hkv,D]` | K/V 分开，每个 `[pages,L,H,P,D]` |
| LM Head | Prefill 取每个所调度请求的末行，partial Chunk 的输出在 postprocess 跳过 | 只选本轮真正完成输入的请求，partial Chunk 可零行 |
| CUDA Graph | 此版本主要捕获 Decode 的 model，LM Head 在捕获外 | 按 N/R 捕获模型、LM Head 与 Argmax，含 Prefill 形状 |
| 生成选择 | Sampler 支持 temperature 路径 | 当前正式路径为 Greedy |

例如 ref_count=0 在 nano 中不一定意味着旧 KV 已被擦掉，只意味着没有活跃使用者；页号再次
分配时才清除相应旧 hash 记录。在 C++ 项目中 cache-only=1，因此只有删除缓存引用后才进入
普通 free list。读别人的缓存代码时，先确认计数定义再讨论驱逐。

## 8. 你现在可以怎样进入 vLLM V1

先把官方组件当作职责地图：请求接收/输入处理、调度和 KV 管理、模型执行、输出处理。
Worker 与 ModelRunner 承担设备执行及输入准备等职责；具体进程拆分比本项目丰富。
参见 [vLLM 官方架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)。

然后聚焦一个概念：V1 的调度表达可以统一为“请求 ID → 本轮 Token 数”，以此覆盖不同阶段的
计算额度。理解这个协议，再去看分支和策略，比先记住完整类继承树更有用。
参见 [vLLM V1 指南](https://docs.vllm.ai/en/latest/usage/v1_guide/)。

最后读前缀缓存的设计说明，比较它的 Block Pool、请求页映射、命中查找与分配过程。
本项目的简化实现用于练习不变量，生产实现还会考虑模型变体、额外缓存身份信息等。
参见 [官方 Prefix Caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)。

建议每次只带一个已会手算的场景去读，例如“已有 16 Token 前缀，来了 18 Token Prompt”。
依次找到查命中、分配剩余容量、返回调度数和执行输入构造，先不追与该场景无关的分支。

## 9. 多卡概念先按“拆什么”区分

| 方式 | 拆分对象 | 一次请求怎样执行 | 本项目状态 |
| --- | --- | --- | --- |
| DP，数据并行 | 不同请求 | 不同副本各自完成请求 | 未做两卡完整请求副本 Benchmark |
| TP，张量并行 | 层内权重/张量计算 | 多卡共同完成层内计算，需要相应通信 | 未实现 |
| PP，流水线并行 | 模型层 | 请求激活在不同层的设备间流动 | 未实现 |
| PD，Prefill/Decode 分离 | 推理阶段 | P 处理 Prompt，迁移 KV 后由 D 继续输出 | 功能版已实现 |

这张表用于概念区分，实际系统可以组合这些方式。不要因为本地 nano 代码导入 torch.distributed
就认为它实现的也是 PD；要看它究竟划分了哪部分权重、请求或阶段。

官方 PD 文档强调分别调节首 Token 与后续 Token 延迟、控制 Prefill 对 Decode 尾延迟的干扰。
因此不能把“开启 PD”直接等同于吞吐提高。
参见 [官方 Disaggregated Prefilling 说明](https://docs.vllm.ai/en/latest/features/disagg_prefill/)。

本项目两卡各放完整权重、独立页池，通过主机中转 K/V；本次短请求实测更慢。理解首 Token
与 KV 的生命周期后，再读 [任务 11](../task_11_pd_disaggregation_zh.md) 的 try_handoff，
就能把这一功能放回已经掌握的 Sequence 与分页知识中。

## 10. 这一节的过关任务

从 nano-vLLM 的 `self.attn(q,k,v)` 出发，向上找到设置 slot_mapping 的函数，向下找到写缓存
的调用。再回到 C++ 项目指出同样的生产者和消费者。

最后用自己的话回答：为什么你熟悉 PyTorch Module，并不意味着已经掌握请求调度；又为什么
掌握 Sequence、元数据和缓存之后，看 nano 的 Python 会明显容易很多。
