# 面试手册 01：固定版本 vLLM 源码对照

[面试学习目录](README.md) · 下一篇：[项目口述与证据](02_project_defense_zh.md)

目标是把“理解自己的项目”转化为“能准确阅读并解释 vLLM 的同类模块”。每节先背结论，再读代码确认条件，最后用追问检查是否真正理解。

## 1. 先记住版本与阅读范围

本篇固定 vLLM **v0.10.2**，提交 **`01efc7ef781391e744ed08c3292817a773d654e6`**，核对日期 2026-09-11。它是选定的学习版本，不代表当前最新版本。所有源码链接都固定到提交，不随 main 更新。

本项目对应源码基线为 `2dfd659`，nano-vLLM 对照固定为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。三者行为不同，本篇不会把相似类名当作相同协议。

本次逐函数讨论先限定为：decoder-only、普通 full attention、无投机解码、无多模态、无 LoRA、无 DCP 的基本生成路径。开启额外功能时，需要重新检查元数据、采样与容量条件。

源码版本与文件哈希保存在 [sources.json](references/sources.json)。vLLM 源码摘录版权属于 vLLM 项目贡献者，按 [Apache License 2.0](references/LICENSE.vllm) 使用；中文讲解与标注的教学伪代码用于说明其逻辑。

本篇只要求读源码。该固定版本 [pyproject.toml](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/pyproject.toml) 声明 Python `>=3.9,<3.14`，本项目 `zyf1` 是 Python 3.8，不能直接把两套依赖混装。这里不通过安装 vLLM 来验证运行兼容性。

**面试 30 秒短答：**

> 我用固定版本 vLLM V1 对照了请求调度、KV 管理和 GPU 输入准备。自己的项目用 C++ 显式连接这些模块，vLLM 则把引擎核心、执行器、Worker 和模型运行器进一步分层。我重点比较职责、数据协议和状态更新时间，而不是只对应类名。

**关键词：** 固定提交、职责对照、协议差异、阅读范围。

## 2. 一次请求的总调用链

建议先把下面的骨架讲熟，再逐个打开文件：

```text
LLMEngine.step
  → EngineCoreClient.get_output（具体 client 决定本地/进程通信路径）
  → EngineCore.step 的基本同步路径
      → Scheduler.schedule
      → Executor.execute_model
          → GPU Worker.execute_model
              → GPUModelRunner.execute_model
                  → _update_states / _prepare_inputs
                  → set_forward_context
                  → model.forward
                  → 选择 hidden 行 → compute_logits → sampler
      → Scheduler.update_from_output
  → 输出处理：组织请求结果
```

它是所选基本路径的职责图，不表示所有节点都在一个进程中直接同步调用；该版本还存在排队执行等其他路径。

源码：[vllm/v1/engine/core.py，第 291—296 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core.py#L291)。

```python
        scheduler_output = self.scheduler.schedule()
        model_output = self.execute_model_with_error_logging(
            self.model_executor.execute_model,  # type: ignore
            scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore
```

本项目的 [GPT2CudaEngine::step](../../mini_vllm/cuda/gpt2_cuda_engine.hpp) 则直接调用 `schedule → run → commit`。相同的是把计划、执行和结果更新分开；不同的是执行边界、进程协议和更多状态处理。

| 你要找的问题 | vLLM 固定源码 | 本项目对应 |
| --- | --- | --- |
| 用户请求怎样推进 | [LLMEngine.step](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/llm_engine.py#L240) | Engine 外层循环 |
| 本轮调度与执行衔接 | [EngineCore.step](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/engine/core.py#L280) | GPT2CudaEngine::step |
| 设备执行入口 | [Worker.execute_model](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_worker.py#L424) | Runner 外层设备入口 |
| 模型前后处理 | [GPUModelRunner.execute_model](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L2000) | Impl::run / forward |
| 请求结果落回状态 | [update_from_output](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/sched/scheduler.py#L862) | Scheduler::commit |

**追问：为什么不把全部逻辑放进 model.forward？**

回答：模型前向消费本轮 Tensor 和缓存；请求的排队、预算、生命周期跨越多次前向。把它们分开，才能在不同请求之间复用执行资源，并独立验证调度协议。

## 3. Scheduler 调度的到底是什么

**30 秒短答：**

> V1 的核心计划可以理解为每个请求本轮需要计算多少个新位置，再受总 Token 预算和 KV 容量约束。Prefill 与 Decode 在这个层面都可以表达成位置数差额；但具体源码还考虑投机 Token、输出占位和模型长度等条件。

源码：[vllm/v1/core/sched/scheduler.py，第 211—218 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/sched/scheduler.py#L211)。

```python
            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)
```

本篇关闭额外功能后，`num_tokens_with_spec` 退化为已知 Prompt 与输出 ID 数，`num_output_placeholders` 不贡献额外位置，主要差额接近本项目 `pending_tokens()`。

**不要直接背“永远等于 tokens−computed”。** 上面真实代码比这个教学简式多了字段；面试遇到投机解码或异步调度时，要承认本篇限定范围。

SchedulerOutput 的关键协议：

源码：[vllm/v1/core/sched/output.py，第 129—134 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/sched/output.py#L129)。

```python
    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
```

区别于 C++ 的 `vector<ScheduledItem>`，这里主要使用请求 ID 到调度数的映射，还区分新请求完整信息与缓存请求的增量信息，见同文件 `scheduled_new_reqs` 和 `scheduled_cached_reqs`。

**小题：** 三个请求本轮调度数 `{A:1,B:7,C:2}`，主体模型处理多少行？

答案：N=10；请求数是 3。不能把最大并发请求数当成矩阵第一维，也不能把 N 当成本轮对外生成数。

**追问：running 优先是不是严格 Decode 优先？**

回答：不一定，running 也可能是上轮尚未完成的 Prompt。需要读运行队列顺序和预算分配，不能仅凭队列名字判断请求阶段。

## 4. 同名 computed 字段，更新时机并不一样

这是本篇最值得记住的差异之一。

源码：[vllm/v1/core/sched/scheduler.py，第 643—646 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/sched/scheduler.py#L643)。

```python
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
```

这几行位于 `_update_after_schedule`，由 `schedule` 在返回计划之前调用。vLLM 会先构建包含所需输入状态的调度输出，再推进调度侧计数；结果阶段还可能根据投机 Token 拒绝等情况回调计数。

本项目则在 Runner 同步返回之后，由 [Scheduler::commit](../../mini_vllm/scheduler.hpp#L93) 执行 `mark_computed`。

| 时间点 | 本项目同步路径 | 本篇 vLLM 调度路径 |
| --- | --- | --- |
| 刚产生本轮计划 | Sequence computed 尚未加本轮数 | 调度输出先保存需要的状态 |
| schedule 返回前 | 仍未提交计算数 | `_update_after_schedule` 已推进计数 |
| 执行结果回来 | commit 加计数并追加样本 | 更新输出、停止状态及必要的计数修正 |

**30 秒短答：**

> computed 的概念是输入处理进度，但读取时必须知道它属于哪个对象和哪个阶段。我的同步 Engine 在执行完成后提交；所选 vLLM 版本会预先推进调度侧进度，并通过输出协议与执行侧状态衔接。因此看见 scheduler 的计数增加，不能单独推断 GPU 已经完成。

**追问：为什么不把本项目也直接提前加？**

回答：当前 ModelInput 从 Sequence 的 computed 计算输入位置。若只把加法前移而不保存旧位置，就会跳过待算 Token。改更新时间必须同时修改数据协议，不能只移动一行代码。

## 5. 缓存命中后，哪些 Token 可以跳过

读 [KVCacheManager.get_computed_blocks](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/kv_cache_manager.py#L153)，它先处理禁用缓存和需要 Prompt logprobs 的路径，再查询完整块命中。

源码：[vllm/v1/core/kv_cache_manager.py，第 179—182 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/kv_cache_manager.py#L179)。

```python
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))
```

`request.num_tokens−1` 保证至少留下一个输入位置来产生 logits，块对齐限制有时会让整块重算。这与本项目保留最后一段 Prompt 的动机相近。

以 block_size=16、已有公共前缀 16、目标 Prompt=18 为例：

```text
命中前：已知 18，已计算 0
查命中：找到前 16 个位置对应的完整 KV 块
分配：保留共享块，为剩余两个位置准备可写空间
执行：输入 position 16、17，context 17、18
采样：用 position 17 的 hidden 预测首输出
```

注意 block_size=16 是本例配置，不是所有 vLLM 后端固定常量。复杂 KV group、滑窗、多模态等情形还需读 coordinator 的具体实现。

**追问：Prompt 相同但要求每个 Prompt Token 的 logprob，还能直接全部跳过吗？**

回答：缓存 K/V 不等于缓存全部 Prompt logits。该版本入口会对需要 Prompt logprobs 的请求跳过此缓存命中路径；应按所需输出和具体版本判断。

## 6. BlockPool 的 ref=0 为什么仍可能缓存有效内容

源码：[vllm/v1/core/block_pool.py，第 244—250 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/block_pool.py#L244)。

```python
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1
```

在 vLLM 这套约定下，没有活跃请求引用的缓存页可以 ref=0 并位于 free queue。命中后，touch 将它移出可回收队列并加引用；重新分配旧页时才通过 `_maybe_evict_cached_block` 去掉旧缓存身份。

本项目缓存自身持有一次引用，所以 cache-only 页 ref=1，删除缓存引用后才进入普通 free list。

| 状态 | 本项目 ref | vLLM 普通非 null 缓存页 ref |
| --- | ---: | ---: |
| 仅缓存保留 | 1 | 0 |
| 缓存 + 一个活跃请求 | 2 | 1 |
| 缓存 + 两个活跃请求 | 3 | 2 |

**背诵句：** 引用数要先问“谁算作一个拥有者”，不能跨项目比较裸数字。

**追问：缓存页在 free queue 里，是不是数据已经清零？**

回答：不是；free queue 表示可回收候选，旧内容和 hash 可能暂时仍然有效。是否可命中、是否被重新分配，取决于缓存索引与引用更新协议。

## 7. Prefix key 为什么还带模型相关身份

读 [hash_block_tokens](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/kv_cache_utils.py#L539)：

源码：[vllm/v1/core/kv_cache_utils.py，第 562—565 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/kv_cache_utils.py#L562)。

```python
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```

hash 包含父块 hash、当前块 Token 和额外身份，避免把不同历史下字面相同的一块混为一谈。

同文件 [generate_block_hash_extra_keys](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/kv_cache_utils.py#L509) 还处理 LoRA、多模态与 cache salt 等额外信息。只比较 Token ID 不能代表所有配置下的等价计算。

本项目固定模型、纯文本、单 Engine，以完整 Token 前缀向量作为 key，容易验证，但不具备这些额外身份管理和哈希效率设计。

**追问：能否把项目的 vector key 直接换成当前块 hash？**

回答：不可以丢掉历史依赖。至少需要父前缀身份，并讨论碰撞和配置身份；“换一个更快容器”不是完整设计。

另一个差异：vLLM `allocate_slots` 的本地路径可以登记计划覆盖的可缓存块，代码位置见 [缓存登记](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/kv_cache_manager.py#L290)。本项目在同步 commit 后登记。读取索引存在与数值就绪的关系时，要结合执行顺序与远端延迟登记协议，不能只看函数名 `cache_blocks`。

## 8. GPUModelRunner 怎样生成你熟悉的 Tensor

源码：[vllm/v1/worker/gpu_model_runner.py，第 883—902 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L883)。

```python
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)
```

三个主要中间量：

1. `num_scheduled_tokens`：每个请求本轮长度。
2. `req_indices`：展开后每个 Token 行属于哪个请求槽位。
3. `positions_np`：对应请求已有 computed 加上本轮局部 offset。

请求槽位顺序来自 `input_batch.req_ids`，因此不能假定字典插入顺序就是最终 Tensor 行顺序。

教学输入：A computed=17，本轮 1；B computed=0，本轮 3：

```text
num_scheduled_tokens = [1,3]
req_indices          = [0,1,1,1]
局部 arange          = [0,0,1,2]
positions            = [17,0,1,2]
query_start_loc      = [0,1,4]
```

本项目 `prepare_packed_model_input` 通过 C++ 双重循环构造相同角色的数组。vLLM 使用 NumPy/Torch 批量索引及持久输入缓冲；实现手段不同，元数据含义相近。

**追问：position 与 packed 行号有什么区别？**

回答：position 是请求内部的绝对位置，影响位置编码和 KV 槽；packed 行号只是本轮矩阵的行坐标，可以随着调度变化。

## 9. 在真实源码里找到物理槽公式

源码：[vllm/v1/worker/block_table.py，第 123—129 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/block_table.py#L123)。

```python
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // self.block_size)
            block_numbers = self.block_table_np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(block_numbers * self.block_size,
                   block_offsets,
                   out=self.slot_mapping_np[:req_indices.shape[0]])
```

这段是 `dcp_world_size==1` 的普通分支：先找请求在页表矩阵中的行，再选逻辑块，得到物理页号，最后加页内 offset。

```text
logical_block = position // block_size
physical_block = block_table[request_slot, logical_block]
slot = physical_block * block_size + position % block_size
```

它与本项目公式一致。vLLM 另有 DCP 分支，需要虚拟块和分布规则；本篇不把普通分支结论推广到那个路径。

**现场手算：** block=16，页表 `[5,2]`，position=17，则 slot=33，context=18。三个数字都能解释，才算记住这个公式。

## 10. 模型的 Attention 参数只有 q/k/v，页表从哪来

从熟悉的 [LlamaAttention.forward](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/model_executor/models/llama.py#L208) 进入：

源码：[vllm/model_executor/models/llama.py，第 213—218 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/model_executor/models/llama.py#L213)。

```python
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
```

对照 PyTorch：QKV projection、按 Q/KV 大小切分、RoPE、Attention、输出投影。Llama 与项目 GPT-2 结构不同，这里比较数据流职责，不比较同 Token 的输出。

继续跟到 [Attention.forward](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/attention/layer.py#L223)，会看到通过 `get_forward_context()` 读取本轮 attention metadata。上游 [GPUModelRunner](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L2054) 用 `set_forward_context` 建立该上下文。

```text
Scheduler 选位置数
  → Runner 生成元数据
  → set_forward_context
  → 普通模型 Module
  → Attention 从 context 找长度、页表和后端所需信息
```

**30 秒短答：**

> vLLM 模型保留普通 PyTorch Module 组合，额外的执行元数据由 Runner 准备并通过 forward context 传递。这样 Attention 能读取分页缓存信息，而模型每层的显式参数不必携带完整调度对象。这里的 context 不是 autograd.Function 用于保存 backward 数据的 ctx。

**追问：这与你的 torch.compile 经验有什么联系？**

回答：我可以沿 Module、算子与编译执行路径定位模型计算；另外还需要沿 Runner 找到本轮 Tensor 与元数据的生产者。请求调度状态跨越多个前向，不能只看一张计算图。

## 11. 采样行选择：本项目与 vLLM 不是完全一致

该版本无投机路径先选择每个已调度请求片段的最后一行：

源码：[vllm/v1/worker/gpu_model_runner.py，第 967—969 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L967)。

```python
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
```

随后选择 hidden 再投影：

源码：[vllm/v1/worker/gpu_model_runner.py，第 2101—2102 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L2101)。

```python
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
```

但其中可能包含还没完成整个 Prompt 的片段。该版本会在 [输出处理](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L1887) 检查 `seq_len < req_state.num_tokens`，丢弃这些 partial prefill 样本，并处理相应随机数状态。

本项目任务 09 更早过滤：只有 `computed+scheduled==num_tokens` 才进入 Gather，因此全是 partial chunk 的一轮可 R=0，不执行 LM head。

**面试短答：**

> 两者都有先取 hidden 行再做词表投影的思路。区别是我项目进一步按输入是否完成过滤行，固定 vLLM 版本的普通分支先取每个片段末行，再丢弃 partial prefill 样本。我不能据此说项目整体比 vLLM 更快，因为实现范围、模型和其他优化完全不同。

**追问：可以删掉 Prompt 中间位置的 Transformer 计算吗？**

回答：不可以，那些位置的 KV 仍被后续 Attention 读取；裁剪边界在末尾词表投影，不是整个模型输入。

## 12. CUDA Graph 为什么不能直接照搬 (N,R) 缓存键

在所选版本 [execute_model](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L2043) 中，先构造 BatchDescriptor，再经 dispatcher 选择 Graph runtime mode，并传给 forward context。

源码：[vllm/v1/worker/gpu_model_runner.py，第 2043—2050 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_model_runner.py#L2043)。

```python
            uniform_decode = (max_query_len
                              == self.uniform_decode_query_len) and (
                                  num_scheduled_tokens
                                  == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = \
                self.cudagraph_dispatcher.dispatch(batch_descriptor)
```

本项目是在一个固定配置 Runner 内按精确 `(N,R)` 缓存直接捕获的设备执行序列；vLLM 的形状组织、执行模式和模型封装更复杂。不能把本项目的 map key 当成 vLLM 所有模式的统一规则。

**30 秒短答：**

> CUDA Graph 的共同前提是执行计划和设备地址满足重放约束，动态内容写进持久缓冲。具体怎样填充形状、选执行模式和决定捕获边界是实现相关的。我能解释项目的 `(N,R)`，对 vLLM 则会沿 BatchDescriptor 和 dispatcher 检查当前版本。

本篇不声称已经验证该 vLLM 版本在本机器上所有 Graph 模式可运行；这里的证据是固定源码阅读。

## 13. 显存不足：缓存淘汰与请求抢占是两件事

vLLM 调度中的请求抢占片段：

源码：[vllm/v1/core/sched/scheduler.py，第 271—280 行](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/core/sched/scheduler.py#L271)。

```python
                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
```

完整分支还会根据策略选牺牲请求；该基本路径释放请求资源、重置 computed、重新放到 waiting，后续重新准备可用缓存并恢复计算。

本项目单卡只淘汰 cache-only 页，不抢走活跃请求 KV；无进展时报错。PD 用接纳前预留最大 Decode 页数和 pending 背压简化活性问题，不能称为 vLLM 式请求抢占。

**背诵句：** 缓存淘汰处理没人正在用的缓存；请求抢占暂停正在运行的工作并承担恢复成本。

**追问：重置 computed 是否意味着所有前缀一定重算？**

回答：不能直接推断，后续仍可能重新命中保留的缓存。重置的是该请求当前的进度状态，实际重算量还取决于恢复时的缓存命中和资源情况。

## 14. 两分钟对照口述

> 我先用 C++ 项目掌握请求的 Token 记账、分页寻址、调度预算和 GPU 执行，再对照固定版本 vLLM V1。两者都把调度、模型执行和结果处理分开。V1 用请求 ID 到本轮 Token 数的映射组织工作，GPUModelRunner 再生成 packed 输入、绝对位置和物理槽映射。
>
> 我重点核对了差异。例如项目的 computed 在同步执行后提交，vLLM 调度侧会预先推进；项目缓存自己持有一个引用，而 vLLM 的空闲缓存页可以 ref=0；项目只投影真正能采样的行，所选 vLLM 普通路径会先取每个片段末行，再丢弃 partial prefill 输出。
>
> 模型计算部分仍是熟悉的 PyTorch Module，但 Attention 通过 forward context 获得运行元数据。我的项目只覆盖小模型、固定范围的功能验证，尚未实现生产引擎里的请求抢占和复杂执行协议。因此我把它作为理解这些不变量和性能验证方法的实践，而不会宣称完整复现 vLLM。

## 15. 闭卷检查：答案不能只报函数名

| 题目 | 必须说出的因果关系 |
| --- | --- |
| 为什么调度单位是 Token 数 | 不同请求阶段都能表达成要推进的输入位置 |
| 为什么 computed 更新后 GPU 未必已完成 | 调度侧可提前记账，执行结果有独立协议 |
| 为什么 ref=0 还能命中 | 没活跃引用不等于旧内容已经被覆盖 |
| 为什么 Prefix key 链接父块 | 同一当前块在不同历史下 KV 不同 |
| 为什么 slot 与 position 不相等 | 逻辑位置经页表映射到非连续物理地址 |
| 为什么 partial 样本不能输出 | Prompt 尚未读完，末行分布不是整个输入的续写 |
| 为什么不能直接比较本项目和 vLLM 速度 | 模型、后端、功能范围与测量条件未对齐 |

过关标准：任选一题，先答 30 秒，再打开对应源码指出生产者、消费者和一个错误边界。下一篇把这些能力组织成项目介绍与证据链。
