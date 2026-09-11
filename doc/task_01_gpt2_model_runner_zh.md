# 开发任务 01：接通 Scheduler 与 GPT2ModelRunner

这篇从你熟悉的 PyTorch `model(input_ids)` 出发，解释一个请求如何经过调度、分页、模型执行，最后生成 Token。先读这一版源码精读，再看文末的开发记录。

学习目标：能够独立跟踪一次 `engine.step()`，准确说出“哪些 Token 已经拥有 KV”“哪些只是生成了 ID”“什么时候可以释放页”。建议分两次阅读：第一次到第 5 节，第二次看提交、测试与练习。

## 1. 先把 Python 中隐含的状态写出来

在普通 PyTorch 生成循环里，你可能这样写：

```python
# 教学伪代码，不是本项目的 Python API。
tokens = prompt
cache = None
for _ in range(max_new_tokens):
    logits, cache = model(tokens if cache is None else tokens[-1:], cache)
    tokens = tokens + [logits[-1].argmax()]
```

这里至少有四种状态，C++ 引擎把它们拆开管理：

| Python 中的对象 | 本项目对象 | 谁负责修改 |
| --- | --- | --- |
| `tokens` 与生成停止条件 | `Sequence` | `Scheduler::commit` |
| 哪些请求进入本轮 `model` | `SchedulerOutput` | `Scheduler::schedule` |
| `past_key_values` 的存储空间 | `BlockManager` + `KVCachePool` | 前者管页号，后者保存数值 |
| 前向中间 Tensor | `GPT2InferenceWorkspace` | Runner 持有，前向覆盖写入 |

`BlockManager` 不是一个 Attention 算子；它不知道 Q、K、V 的数值。`KVCachePool` 也不负责选择哪个请求优先运行。先把控制状态与数值计算分开，后面的调用就容易理解。

当前任务走 **CPU Runner**。任务 04、05 的 GPU Runner 保留相同调度思想，但一次执行 packed Token；不要把 CPU 微步循环当作当前 GPU 执行方式。

## 2. 阅读入口与调用链

按下表顺序打开文件，不必从头读完 `train_gpt2.c`。

| 顺序 | 打开位置 | 读完要回答的问题 |
| --- | --- | --- |
| 1 | [请求状态](../mini_vllm/sequence.hpp#L21) | `computed` 与 `tokens.size()` 为什么不同？ |
| 2 | [Engine 入口](../mini_vllm/gpt2_engine.hpp#L43) | 谁接收请求，谁推动一轮执行？ |
| 3 | [调度选择](../mini_vllm/scheduler.hpp#L56) | 一轮每个请求计算多少 Token？ |
| 4 | [CPU Runner](../mini_vllm/gpt2_model_runner.hpp#L54) | 多 Token 计划怎样分成微步？ |
| 5 | [输入元数据](../mini_vllm/model_input.hpp#L112) | 行号如何对应请求与物理页？ |
| 6 | [推理工作区](../train_gpt2.cpp#L38) | 哪些内存跨步保留？ |
| 7 | [真实模型测试](../dev/test_gpt2_engine.cpp) | 哪个断言能发现页映射或生成错误？ |

```text
调用方：用户循环 / gpt2_engine_demo
  └─ GPT2Engine::step
      ├─ Scheduler::schedule
      │   └─ try_schedule → BlockManager::ensure_capacity
      ├─ GPT2ModelRunner::run
      │   └─ 对 micro_step 循环
      │       ├─ prepare_model_input
      │       └─ gpt2_forward_inference_with_workspace
      └─ Scheduler::commit
          ├─ mark_computed
          ├─ append_token（仅输入全部处理完时）
          └─ release（请求完成时）
```

这里有两个循环：外层 `engine.step` 是服务调度轮；内层 `micro_step` 是 CPU 模型执行轮。一次调度不是一次单 Token 前向。

## 3. 请求进入系统时，为什么容量是 P + G − 1

源码：[mini_vllm/gpt2_engine.hpp，第 43—59 行](../mini_vllm/gpt2_engine.hpp#L43)。以下为当前文件的原样摘录。

```cpp
    std::shared_ptr<Sequence> add_request(
        std::uint64_t request_id, std::vector<int> prompt_tokens,
        SamplingParams sampling_params) {
        if (prompt_tokens.empty()) {
            throw std::invalid_argument("request prompt must not be empty");
        }
        if (sampling_params.max_new_tokens == 0) {
            throw std::invalid_argument("max_new_tokens must be positive");
        }
        const std::size_t additional_processed_tokens =
            sampling_params.max_new_tokens - 1;
        if (prompt_tokens.size() > max_context_length_ ||
            additional_processed_tokens >
                max_context_length_ - prompt_tokens.size()) {
            throw std::out_of_range(
                "request can exceed the configured context capacity");
        }
```

逐段看这段入口检查：

1. 空 Prompt 没有可以执行的最后一行，无法产生第一个预测，因此拒绝。
2. `max_new_tokens == 0` 在本接口不代表“仅建立缓存”，因此拒绝。
3. Prompt 长度记为 `P`，生成数记为 `G`。处理完 Prompt 后已经能采样第 1 个输出，只需再执行 `G−1` 个输入 Token。
4. 最后一个生成的 Token ID 返回给用户后，请求就结束；无需再为它计算 KV。
5. 检查用减法表达剩余容量，并先检查 Prompt 是否越界，避免无符号加法溢出或减法下溢。

手算：`P=17, G=4`，最终 `token_ids.size()=21`，实际最多计算 20 个 Token。页大小 16，最大需要 `ceil(20/16)=2` 页，不是按 21 次模型输入收费。

这与训练中“每个位置都有 label 和 loss”的理解不同。这里输出目标是生成 ID，不是把最后一个生成位置继续送进模型。

## 4. Engine 如何把三个阶段接起来

源码：[mini_vllm/gpt2_engine.hpp，第 77—104 行](../mini_vllm/gpt2_engine.hpp#L77)。以下为当前文件的原样摘录。

```cpp
        EngineStepResult result;
        result.free_blocks_before_schedule =
            block_manager_.num_free_blocks();
        SchedulerOutput output = scheduler_.schedule();
        result.free_blocks_after_schedule =
            block_manager_.num_free_blocks();
        if (output.items.empty()) {
            throw std::runtime_error(
                "scheduler made no progress; KV cache may be exhausted");
        }

        result.num_batched_tokens = output.num_batched_tokens;
        for (const ScheduledItem& item : output.items) {
            result.request_ids.push_back(item.sequence->request_id());
            result.phases.push_back(item.phase);
            result.scheduled_token_counts.push_back(
                item.num_scheduled_tokens);
            result.block_tables_before_commit.push_back(
                item.sequence->block_table());
        }

        result.sampled_token_ids = model_runner_.run(output);
        result.num_micro_batches =
            model_runner_.last_model_inputs().size();
        scheduler_.commit(output, result.sampled_token_ids);
        result.free_blocks_after_commit =
            block_manager_.num_free_blocks();
        return result;
```

`free_blocks_before_schedule` 是分配前的状态；`free_blocks_after_schedule` 是为本轮输入准备好页之后的状态；`free_blocks_after_commit` 可能因为请求结束而回升。

三个关键调用的副作用不同：

| 调用 | 主要写入 | 此时可以认为 Token 已完成吗？ |
| --- | --- | --- |
| `schedule()` | 队列、请求页表、调度计划 | 不能，只是保证空间与选择输入 |
| `run(output)` | KV 数值、临时激活、采样结果 | 数值已完成，Sequence 计数尚未提交 |
| `commit(...)` | computed、生成 ID、结束状态、页引用 | 可以，对外状态正式前进 |

为什么把 `block_tables_before_commit` 复制到结果里？结束请求的 `release` 会清空页表。如果等提交后再记录，你看到的是空表，无法还原模型本轮真正用过的物理页。

注意这是正常执行路径的阶段划分，不是完整数据库事务；异常恢复、请求抢占、重试并没有因为这个顺序自动实现。空调度会抛错，避免引擎无进展地死循环。

## 5. CPU 微步：行号会变化，请求身份不能丢

源码：[mini_vllm/gpt2_model_runner.hpp，第 64—93 行](../mini_vllm/gpt2_model_runner.hpp#L64)。以下为当前文件的原样摘录。

```cpp
        std::size_t max_micro_steps = 0;
        for (const ScheduledItem& item : output.items) {
            if (item.sequence == nullptr || item.num_scheduled_tokens == 0) {
                throw std::invalid_argument("scheduled item is invalid");
            }
            max_micro_steps =
                std::max(max_micro_steps, item.num_scheduled_tokens);
        }

        std::vector<int> sampled_token_ids(output.items.size(), -1);
        last_model_inputs_.clear();
        last_model_inputs_.reserve(max_micro_steps);

        for (std::size_t micro_step = 0; micro_step < max_micro_steps;
             ++micro_step) {
            ModelInput input = mini_vllm::prepare_model_input(
                output, micro_step, block_manager_, max_context_length_,
                max_blocks_per_sequence_,
                static_cast<std::size_t>(kv_cache_pool_.num_pages));
            PageTable page_table(
                checked_int(input.batch_size(), "micro batch is too large"),
                checked_int(max_blocks_per_sequence_,
                            "page table width is too large"));
            page_table.context_lengths = input.context_lengths;
            page_table.block_tables = input.block_tables;

            gpt2_forward_inference_with_workspace(
                &model_, input.token_ids.data(), &kv_cache_pool_, &page_table,
                checked_int(input.batch_size(), "micro batch is too large"),
                &workspace_);
```

假设调度结果是 `A:3 tokens, B:1 token`：

| micro_step | 本次模型行 | 行数 | 对应原调度 item |
| ---: | --- | ---: | --- |
| 0 | A 的第 0 个待算 Token，B 的第 0 个待算 Token | 2 | `[0,1]` |
| 1 | A 的第 1 个待算 Token | 1 | `[0]` |
| 2 | A 的第 2 个待算 Token | 1 | `[0]` |

`max_micro_steps` 是最大计划长度 3，不是计划总长度 4。每个微步可以批处理多个请求；长度不足的请求会跳过。

下一段负责把请求状态变成模型输入：

源码：[mini_vllm/model_input.hpp，第 145—166 行](../mini_vllm/model_input.hpp#L145)。以下为当前文件的原样摘录。

```cpp
        input.token_ids.push_back(sequence.token_ids()[position]);
        input.positions.push_back(model_input_checked_int(
            position, "token position is too large"));
        input.context_lengths.push_back(model_input_checked_int(
            position + 1, "context length is too large"));
        const std::size_t physical_slot =
            static_cast<std::size_t>(physical_block) *
                block_manager.block_size() +
            block_manager.slot_for_token(position);
        input.slot_mapping.push_back(model_input_checked_int(
            physical_slot, "physical KV slot is too large"));
        input.request_ids.push_back(sequence.request_id());
        input.scheduled_item_indices.push_back(item_index);

        const std::size_t row_start = input.block_tables.size();
        input.block_tables.resize(
            row_start + max_blocks_per_sequence, -1);
        std::copy(
            sequence.block_table().begin(),
            sequence.block_table().end(),
            input.block_tables.begin() +
                static_cast<std::ptrdiff_t>(row_start));
```

这些数组不是重复信息，每个负责一条索引关系：

- `positions`：绝对位置，GPT-2 位置 Embedding 要用它。
- `context_lengths`：当前 Query 可以看多少个历史位置，含自己。
- `slot_mapping`：当前 K/V 写到哪里，只指向一个物理 Token 槽。
- `block_tables`：历史 K/V 去哪里读，每行是一张完整页表。
- `scheduled_item_indices`：模型第几行属于原调度中的第几个请求。

例如页大小 16，A 的页表为 `[5,2]`，处理位置 17：逻辑页为 1，物理页为 2，页内偏移为 1，写入槽是 `2*16+1=33`。绝对位置仍是 17，不能把 33 传给位置 Embedding。

页表尾部补 `-1` 是为了固定行宽；只有 `context_length` 覆盖的逻辑页才应该被读取。`-1` 不是一个可以读取的“空白物理页”。

## 6. 采样条件为什么检查 position + 1

源码：[mini_vllm/gpt2_model_runner.hpp，第 95—111 行](../mini_vllm/gpt2_model_runner.hpp#L95)。以下为当前文件的原样摘录。

```cpp
            const ActivationTensors& acts = workspace_.acts();
            const int padded_vocab_size = model_.config.padded_vocab_size;
            for (std::size_t row = 0; row < input.batch_size(); ++row) {
                const std::size_t item_index =
                    input.scheduled_item_indices[row];
                const Sequence& sequence =
                    *output.items[item_index].sequence;
                const std::size_t position =
                    static_cast<std::size_t>(input.positions[row]);
                if (position + 1 == sequence.num_tokens()) {
                    sampled_token_ids[item_index] = greedy_argmax(
                        acts.logits + row * padded_vocab_size);
                }
            }
            last_model_inputs_.push_back(std::move(input));
        }
        return sampled_token_ids;
```

这里先从模型行找到 `item_index`，再找到原来的 `Sequence`。不能直接写 `sampled_token_ids[row]`：微步后半段某些请求已退出，本轮行号与原始 item 顺序可能不再一一对应。

`position + 1 == sequence.num_tokens()` 表示当前处理的是**所有已知输入的最后一个位置**。Prompt 中间位置虽然也有 logits，但它预测的是 Prompt 内的下一个位置，不应追加成用户输出。

默认 `sampled_token_ids` 全是 `-1`。这个值是“本轮尚不能采样”的控制标记，不是词表里的特殊 Token。实际 argmax 只遍历真实词表大小，行跨度则用 `padded_vocab_size`；两者职责不同。

## 7. 提交顺序决定状态是否自洽

源码：[mini_vllm/scheduler.hpp，第 93—113 行](../mini_vllm/scheduler.hpp#L93)。以下为当前文件的原样摘录。

```cpp
        for (std::size_t i = 0; i < output.items.size(); ++i) {
            const ScheduledItem& item = output.items[i];
            Sequence& sequence = *item.sequence;
            sequence.mark_computed(item.num_scheduled_tokens);
            block_manager_.cache_computed_prefix_blocks(sequence);
            if (sequence.pending_tokens() != 0) {
                if (sampled_token_ids[i] != -1) {
                    throw std::logic_error("partial prefill must not produce a sampled token");
                }
                continue;
            }
            const int sampled_token = sampled_token_ids[i];
            if (sampled_token < 0) {
                throw std::logic_error("completed model input requires a sampled token");
            }
            sequence.append_token(sampled_token);
            if (sequence.should_finish_after(sampled_token)) {
                sequence.set_status(SequenceStatus::Finished);
                block_manager_.release(sequence);
                finished.push_back(item.sequence);
            }
```

先把本轮算过的数目加到 `computed`，再判断是否还有待算输入。这样同一段逻辑能处理三种情况：

1. Prefill chunk 未到 Prompt 末尾：`pending>0`，必须收到 `-1`。
2. Prefill 最后一块：`pending==0`，追加第一个输出 Token。
3. Decode：处理上一轮输出的那个 Token，随后追加新的输出。

`append_token` 后通常又出现一个待算 Token。这是自回归生成的正常状态。只有下一轮需要继续生成时，才把它送去计算 KV。

完整手算，Prompt 5 个 Token，预算每轮 3，生成 3 个，忽略 EOS：

| 时刻 | 本轮计划 | 提交后 computed | 提交后总 Token 数 | 输出数 | 下一步 |
| --- | ---: | ---: | ---: | ---: | --- |
| 初始 | — | 0 | 5 | 0 | Prefill |
| 第 1 轮 | 3 | 3 | 5 | 0 | Prefill，返回 `-1` |
| 第 2 轮 | 2 | 5 | 6 | 1 | Decode |
| 第 3 轮 | 1 | 6 | 7 | 2 | Decode |
| 第 4 轮 | 1 | 7 | 8 | 3 | Finished，释放页 |

完成时 `computed=7 < total=8` 完全正确。若为了“让两个数相等”再调用一次模型，反而多做了一次无用计算。

## 8. 工作区与 KV 的生命周期

源码：[train_gpt2.cpp，第 48—62 行](../train_gpt2.cpp#L48)。以下为当前文件的原样摘录。

```cpp
        fill_in_activation_sizes(act_sizes_, config, max_batch_size, 1);
        // PagedAttention 使用独立 scratch；greedy decoding 直接对 logits
        // 取 argmax，不需要训练 Attention、概率和 loss 缓冲。
        act_sizes_[6] = 0;   // preatt
        act_sizes_[7] = 0;   // att
        act_sizes_[21] = 0;  // probs
        act_sizes_[22] = 0;  // losses
        for (std::size_t size : act_sizes_) {
            num_activations_ += size;
        }
        attention_scratch_.resize(
            static_cast<std::size_t>(max_batch_size) * config.num_heads *
            max_context_length);
        acts_memory_ = malloc_and_point_activations(&acts_, act_sizes_);
    }
```

你可以把 Workspace 理解为提前分配的若干 `torch.empty`，但它不保存 autograd 图，也不要求每次前向都重新申请。

`fill_in_activation_sizes(..., max_batch_size, 1)` 中的 `1` 是单个微步的 Token 长度。它不限制请求只能有一个 Token；历史长度由 KV Cache 和独立 Attention scratch 表达。

`preatt/att` 的训练缓冲被去掉，greedy 又不需要完整概率和 loss。Attention scratch 仍按 `B*H*max_context` 分配，因此“省去了 T² 缓冲”不等于“Attention 没有临时存储”。

生命周期应这样画：

```text
model 参数：加载一次 ─────────────────── Engine 使用期间
KV Pool：  分配一次 ── 每轮写新位置 ─── 请求结束后对应页可复用
Workspace：分配一次 ── 每次前向覆盖 ─── Runner 析构时释放
ModelInput：          本轮构造 → 使用 → 保存最近调试快照
```

C++ 成员的声明顺序决定构造顺序。Engine 先声明 BlockManager，再 Scheduler，再 Runner，后两者引用前面的对象；析构顺序相反。阅读这些类时，不要只看构造函数参数排列。

## 9. 从断言学习，而不只运行 PASS

打开 [测试的调度与微步断言](../dev/test_gpt2_engine.cpp#L90)，重点看：

- `num_micro_batches`：验证 CPU 的分步数量，不是 GPU kernel 数。
- `mixed_inputs.front().positions`：验证 Decode 的绝对位置不会因新 Prefill 加入而归零。
- 位置 15、16 的 `slot_mapping`：验证跨物理页边界。
- 最后 `num_free_blocks()==num_blocks()`：默认未开启 Prefix Cache 时验证全部页归还。
- 后半段 full-prefix reference：固定已生成前缀，再计算最后位置的参考 logits 与 greedy Token。

推荐先运行不需要权重的小型控制流例子：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
c++ -std=c++17 -O0 -g -gdwarf-4 -I. mini_vllm/demo.cpp -o /tmp/zyf_learning_demo
/tmp/zyf_learning_demo
```

这个 demo 的 Token 是模拟采样结果，只用于观察调度与页管理。验证真实模型数值要使用本篇末尾的 GPT-2 测试命令。

GDB 可在 `scheduler.hpp:96` 暂停，观察 `item.num_scheduled_tokens`、`sequence` 的计数；再单步到 `append_token` 前后。变量被优化掉时用上述 `-O0 -g -gdwarf-4` 构建，避免把调试信息问题误判为逻辑错误。

## 10. 自测题与参考答案

**题 1：A 本轮调度 4 个 Token，B 调度 2 个，CPU 模型会调用几次？**

答案：4 次，行数依次为 2、2、1、1；共处理 6 个输入行。不要把“4 次前向”说成“6 次前向”，也不要把它套到 packed GPU Runner 上。

**题 2：Prompt 长 18，已计算 16，本轮只算 1 个，应该返回什么？**

答案：`-1`。执行位置 16 后还剩位置 17 未处理，当前 logits 不能代表整个 Prompt 的续写分布。

**题 3：页表 `[7,3]`，位置 16 的 position、context、slot 各是多少？**

答案：16、17、48。三者分别表达位置语义、可见长度、物理写入地址，不能互换。

**题 4：所有请求 Finished 后还有一个生成 ID 没有 KV，是泄漏吗？**

答案：不是。最后一个输出不再作为模型输入，KV 页已经可以释放。泄漏应检查页引用与空闲页数量，不能用 `computed==total` 作为结束条件。

**题 5：本轮 Runner 出错后是否保证能直接重试？**

答案：没有这个保证。KV 可能已部分写入，调度也可能已分配页。当前学习重点是正常路径的不变量，完整故障恢复是尚未实现的服务能力。

下一篇：[任务 02：怎样证明一个优化有效](task_02_benchmark_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[生成与 KV Cache](from_pytorch/01_generation_and_kv.md)、[请求与调度](from_pytorch/02_requests_and_scheduler.md)。
先手推 demo 的四轮状态，再阅读本任务的 CPU 模型接入。

**状态：已完成。**

## 为什么这是下一步

项目现在已经有两部分能力：

1. Sequence、BlockManager、Scheduler 能决定“本轮计算哪些请求、计算多少 Token”。
2. GPT-2 增量前向能根据每个请求独立的 context length 读写分页 KV Cache。

本任务已经通过统一 ModelRunner 接口连接这两部分：调度结果会转换为模型输入，模型输出
经过 greedy sampling 后提交回 Scheduler，完成请求状态更新和 Block 回收。

## 需要建立的数据结构

```cpp
struct ModelInput {
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> context_lengths;
    std::vector<int> slot_mapping;
    std::vector<int> block_tables;
    std::vector<std::uint64_t> request_ids;
};
```

字段含义：

- `token_ids`：本轮真正送入模型的 Token。
- `positions`：Token 在所属请求中的绝对位置。
- `context_lengths`：各请求本轮可见的上下文长度。
- `slot_mapping`：新 K/V 写入的物理页号和页内偏移。
- `block_tables`：逻辑页号到物理页号的映射。
- `request_ids`：模型输出与 Sequence 之间的对应关系。

## 实现步骤

### 1. 独立推理工作区

新增只服务于增量推理的 Workspace，至少保存：

- 单 Token 的 residual、LayerNorm、QKV、Attention 和 MLP 中间结果。
- 大小为 `max_active_sequences × num_heads × max_context_length` 的 Attention scratch。
- 大小为 `max_active_sequences × padded_vocab_size` 的 logits。

这样 ModelRunner 不需要先调用训练前向来间接分配激活内存。

### 2. 构造模型输入

ModelRunner 遍历 `SchedulerOutput.items`：

1. 从 Sequence 的 `num_computed_tokens` 找到本轮起始位置。
2. 读取对应的 Token。
3. 计算绝对 position 和 context length。
4. 从 Sequence 的 Block Table 计算物理页及页内 offset。
5. 将不同请求压紧成当前活跃 Batch。

如果某个 Prefill chunk 包含多个 Token，第一版可在 ModelRunner 内按 position 拆成多个
微步。正确性稳定后，再实现一次处理多个 Prefill Token 的专用路径。

### 3. 对齐 BlockManager 与 KVCachePool

BlockManager 分配的 Block ID 必须直接索引 KVCachePool 的物理页。ModelRunner 不再调用
KVCachePool 自己的页分配接口，避免出现两套互不一致的页所有权。

### 4. 执行和采样

每个微步调用 GPT-2 增量前向。只有当 ScheduledItem 已经计算完当前全部输入 Token 时，
才从最后一个位置的 logits 执行 greedy argmax，并返回 sampled token；部分 Prefill
返回 `-1`。

### 5. Engine 循环

```cpp
while (!scheduler.is_finished()) {
    SchedulerOutput output = scheduler.schedule();
    std::vector<int> sampled = model_runner.run(output);
    scheduler.commit(output, sampled);
}
```

这个循环是学习 vLLM 主链路的最小闭环。

## 验收标准

- [x] 两个以上异长请求能够在不同时间加入。
- [x] 同一轮能够同时处理至少一个 Decode 请求和一个 Chunked Prefill 请求。
- [x] 请求完成后 Block 数量立即恢复，后续请求能够复用这些 Block。
- [x] 每个请求生成的 greedy token 与独立完整前缀重算结果一致。
- [x] 覆盖第 16→17 Token 的跨页扩容；既有模型测试覆盖第 32→33 Token。
- [x] Scheduler 无法产生工作时由 Engine 报错，避免静默死循环。
- [x] AddressSanitizer 和 UndefinedBehaviorSanitizer 检查通过。

端到端测试入口为 `dev/test_gpt2_engine.cpp`。测试还验证了独立推理 Workspace：
该场景使用 435,556 个模型激活元素，而 B=3、T=18 的完整前缀 reference 使用
14,756,418 个激活元素。这个数字只描述该固定测试形状，不是通用显存节省比例。

## 后续任务的完成情况

1. 可重复 Benchmark 已完成，报告 TTFT、TPOT、吞吐、P50/P95 和原始 JSON/CSV。
2. CUDA PagedAttention、GPU ModelRunner、Packed Prefill 与 FP16/BF16 已接通。
3. 完整 Block Prefix Cache、引用计数和 LRU 已完成；抢占仍未实现。
4. Residual + LayerNorm 融合与 CUDA Graph 已完成，详见任务 07 学习手册。
