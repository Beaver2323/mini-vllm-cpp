# 开发任务 05：GPU Multi-Token Prefill

你已经会写 PyTorch 前向，本任务只改变“这次前向装进来哪些位置”。核心是把调度输出的多个不等长片段拼成 N 行，同时保留每行所属请求、绝对位置与可见历史。

学习目标：手写一个包含 Decode 和 Prefill 的 ModelInput，并解释为什么 packed 后不会跨请求 Attention，也不会偷看未来位置。

## 1. 三个相似词先分清

| 概念 | 回答的问题 | 本项目对应 |
| --- | --- | --- |
| Continuous batching | 每轮哪些请求一起推进？ | Scheduler 动态选择请求 |
| Chunked prefill | 一个长 Prompt 本轮算多少？ | token budget 限制每请求本轮数量 |
| Packed execution | 选中的片段怎样组织成模型输入？ | 按请求拼接所有本轮 Token 行 |

它们不是同义词。CPU Runner 可以有 continuous batching 和 chunked prefill，却仍在内部逐 Token 微步执行。任务 05 才把 GPU 前向改为本轮全部 Token 一次组织。

这也不是 PyTorch 的 `PackedSequence` RNN 接口；这里只是“去除 padding 后拼接”的数据组织方式。

## 2. 源码阅读顺序与调用链

| 文件与符号 | 重点 |
| --- | --- |
| [Scheduler::try_schedule](../mini_vllm/scheduler.hpp#L123) | 本轮长度从哪里来 |
| [ModelInput 字段](../mini_vllm/model_input.hpp#L17) | N 行与请求数的区别 |
| [prepare_packed_model_input](../mini_vllm/model_input.hpp#L41) | 外层请求循环、内层 Token 循环 |
| [Runner::run](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L740) | 只构造一个 packed input |
| [Attention 可见长度](../mini_vllm/cuda/paged_attention.cu#L121) | 因果约束落在哪个循环上 |
| [GPU 测试](../dev/cuda/test_gpt2_cuda_model_runner.cu) | 混合批次、跨页、输出一致性 |

```text
schedule → items=[A:qA, B:qB, ...]
         → packed N=qA+qB+...
         → 每层对 N 行做 Linear/LN/Attention
         → 找到已完成输入请求的最后一行
         → 每个请求提交自己的 q 与采样结果
```

CPU Runner 的 `max(qA,qB)` 次微步，变成 GPU Runner 的一次 packed forward。这不意味着只发射一次 CUDA kernel：每层仍包含多个 kernel 和 GEMM。

## 3. token budget 如何决定片段长度

源码：[mini_vllm/scheduler.hpp，第 123—143 行](../mini_vllm/scheduler.hpp#L123)。以下为当前文件的原样摘录。

```cpp
    bool try_schedule(const std::shared_ptr<Sequence>& sequence,
                      SchedulerOutput& output) {
        if (sequence->status() == SequenceStatus::Waiting) {
            block_manager_.apply_prefix_cache(*sequence);
        }
        if (sequence->pending_tokens() == 0) {
            throw std::logic_error("sequence has no input token awaiting model execution");
        }
        const std::size_t budget =
            config_.max_num_batched_tokens - output.num_batched_tokens;
        const std::size_t count = std::min(sequence->pending_tokens(), budget);
        const std::size_t target = sequence->num_computed_tokens() + count;
        if (!block_manager_.ensure_capacity(*sequence, target)) {
            return false;
        }
        output.items.push_back(
            {sequence, sequence->is_prefill() ? ExecutionPhase::Prefill
                                              : ExecutionPhase::Decode,
             count});
        output.num_batched_tokens += count;
        return true;
```

预算限制的是 **本轮新增计算的输入 Token 数**，不是上下文总长度，也不是最多输出多少个 Token。

假设 A 已进入 running、还剩 1 个 Decode 输入，B 是新请求、有 20 个 Prompt Token，预算 8。若 A 先被遍历，计划是 `A:1, B:7`，共 8 行；B 只算前 7 个 Prompt，不能采样。

running 队列也可能包含未完成的 Prefill。源码是按 running 顺序优先推进，不能表述成“所有 Decode 永远严格优先于所有 Prefill”。具体延迟行为还取决于队列顺序和预算。

`ensure_capacity` 在计划进入输出前准备好目标位置的页。如果没有页，代码返回 false；当前系统没有自动把已运行请求的 KV 换出并重算的抢占机制。

## 4. query_start_locations 是请求边界，不是 position

源码：[mini_vllm/model_input.hpp，第 49—65 行](../mini_vllm/model_input.hpp#L49)。以下为当前文件的原样摘录。

```cpp
    for (std::size_t item_index = 0;
         item_index < output.items.size(); ++item_index) {
        const ScheduledItem& item = output.items[item_index];
        if (item.sequence == nullptr || item.num_scheduled_tokens == 0) {
            throw std::invalid_argument("scheduled item is invalid");
        }
        const Sequence& sequence = *item.sequence;
        if (sequence.block_table().size() > max_blocks_per_sequence) {
            throw std::out_of_range(
                "sequence block table exceeds ModelRunner capacity");
        }
        input.query_start_locations.push_back(input.batch_size());

        for (std::size_t offset = 0;
             offset < item.num_scheduled_tokens; ++offset) {
            const std::size_t position =
                sequence.num_computed_tokens() + offset;
```

进入每个请求的 Token 循环前，记录“当前已经拼入多少行”。最后再补 N，就得到长度为 `请求数+1` 的边界数组。

例如本轮长度 `[1,3,2]`：

```text
packed 行： 0 | 1 2 3 | 4 5
请求：      A |   B   |  C
qstart：  [0, 1, 4, 6]
```

B 的行区间是 `[qstart[1],qstart[2]) = [1,4)`，最后一行为 3。这个边界只能找出“本轮片段的最后一行”；是否可采样还要检查整个已知输入是否都处理完。

如果 B 原先已计算 10 个 Token，它的 position 是 10、11、12，而 packed 行号仍是 1、2、3。把两者混用会让位置 Embedding 和 KV 写入都出错。

## 5. 每一行都带独立元数据

源码：[mini_vllm/model_input.hpp，第 79—100 行](../mini_vllm/model_input.hpp#L79)。以下为当前文件的原样摘录。

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

本实现给同一请求的每个 Token 行重复一份页表，便于复用任务 03 的“一行一个 Query” Attention 接口。这样教学路径简单，但元数据开销随 N×页表宽度增长。

可以把它想成 PyTorch 中先 `cat` 多个输入片段，再为每行构造一个“属于哪个序列”的描述。单纯 `torch.cat` Token，而不带请求边界、位置和历史映射，无法保证正确 Attention。

`slot_mapping` 是当前行新 KV 的唯一写地址；`block_tables` 是这行读历史的映射。页表相同不意味着写槽相同：同一请求连续 3 个 Token 会写同一物理页里的不同位置。

## 6. 一个完整的混合批次，逐项手算

使用生产代码的页大小 16，设最大页表宽度为 4：

- A 的 Prompt 已完成，现在 `computed=17,total=18`，待算 ID 为 101，页表 `[5,2]`。
- B 是新 Prompt `[201,202,203]`，`computed=0,total=3`，页表 `[7]`。
- 本轮计划 A 算 1 个，B 算 3 个。

| packed 行 | 请求 | token | position | context | slot | 页表（补齐） |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | A | 101 | 17 | 18 | 33 | `[5,2,-1,-1]` |
| 1 | B | 201 | 0 | 1 | 112 | `[7,-1,-1,-1]` |
| 2 | B | 202 | 1 | 2 | 113 | `[7,-1,-1,-1]` |
| 3 | B | 203 | 2 | 3 | 114 | `[7,-1,-1,-1]` |

边界数组是 `[0,1,4]`。A 采样行 0，B 采样行 3。模型完成后得到两个输出 ID，提交时 A 的 computed 增加 1，B 增加 3。

特别注意 B 的行 1 可见长度只有 1。虽然行 2、3 的 K/V 已经写进物理页 7，它们不会被行 1 的 Attention 读取。

如果 B 的 Prompt 实际还有第 4 个 Token，但本轮只算 3 个，上表输入不变，B 的采样资格改变：它应返回 `-1`，任务 09 的采样行列表也只包含 A 的行 0。

## 7. 为什么一个模型层内可以同时写全部新 K/V

对一个标准 causal Transformer 层，当前层的 Q/K/V 来自每个位置的上一层 hidden。该层的 Linear/LN 对 Token 行独立操作，不需要先算完本层前一个位置的 Attention 才能投影下一个位置的 K。

```python
# 单层的教学逻辑。
q, k, v = project(all_scheduled_rows)
write_all_new_kv(k, v, slots)
for row in rows:
    out[row] = attention(q[row], history_up_to(position[row]))
```

同一层的因果性由读取范围维护；层与层之间则按 stream 顺序执行。后一层使用前一层已经完成的 hidden，因此不会绕过 Transformer 的依赖。

这解释了 packed Prefill 为什么数学上成立，也说明它不是把自回归 Decode 的多个未知输出一次猜出来。未来输出 ID 尚未知晓，仍然需要下一轮采样后再执行。

## 8. 为什么可以提升 GEMM 利用率

设 channels=768。CPU 微步里一个请求的 QKV 是 `[1,768]×[768,2304]`；长 Prompt 32 个位置分别调用 32 次，形状很窄。

packed 后可以是 `[32,768]×[768,2304]`。参数权重与数学表达式不变，但一次 GEMM 包含更多行，减少重复的 host 调用与小矩阵开销。

不要从这里推导“32 个 Token 一定快 32 倍”：Attention 仍按每行可见长度访问历史，GEMM 实际算法、显存带宽和 kernel launch 也会影响结果。

最有意义的实测指标是：相同生成输出下总时间是否降低、TTFT/TPOT 是否变化、GPU timeline 中小型重复 GEMM 是否减少。

## 9. 预算不是越大越好

预算增大时可以减少 Prefill 分块轮数，但代价是单轮执行更长，Decode 结果可能等待整轮结束才交付。

| 修改 | 常见潜在收益 | 需要观察的代价 |
| --- | --- | --- |
| budget 变大 | 更大的 GEMM、较少调度轮 | Decode 等待、激活容量增大 |
| budget 变小 | 更细的调度机会 | Prefill 轮数与提交开销增多 |
| max sequences 变大 | 更多请求同时运行 | KV 压力与每轮竞争增加 |

这些是解释实验的方向，不是没有条件的性能保证。用文末 budget sweep 的同一负载验证，先记录实际每轮 N，再解释曲线。

调试时若 `ModelInput.batch_size()` 不等于调度总 Token 数，优先检查拼接循环，而不是调整 CUDA block size。

源码：[mini_vllm/model_input.hpp，第 103—109 行](../mini_vllm/model_input.hpp#L103)。以下为当前文件的原样摘录。

```cpp
    input.query_start_locations.push_back(input.batch_size());
    if (input.batch_size() == 0 ||
        input.batch_size() != output.num_batched_tokens) {
        throw std::logic_error(
            "packed ModelInput does not match scheduled token count");
    }
    return input;
```

这个断言验证调度计划没有在数据组织过程中被遗漏或重复。它不能证明每一行的 position 和 slot 都正确，仍需要上面的手算与参考测试。

## 10. 练习入口与答案

先运行 [PyTorch packed 演示](from_pytorch/examples/attention_and_pages.py)：

```bash
conda run -p /home/miniconda3/envs/zyf1 python \
  doc/from_pytorch/examples/attention_and_pages.py packed
```

它用小 Tensor 检查按采样行选择 hidden 与投影 logits 的对应关系。生产代码页大小 16，演示页大小 4，迁移手算时不要照抄演示 slot。

**题 1：本轮请求数 3，计划长度 `[2,1,5]`，N 与 qstart 是多少？**

答案：N=8，qstart=`[0,2,3,8]`；各片段最后行为 1、2、7。

**题 2：Prompt 长 10，computed=4，本轮调度 3，最后行能产生输出吗？**

答案：不能，算到 computed=7 后还缺 3 个 Prompt 输入。末行有 logits，不等于这行应对外采样。

**题 3：两个请求 packed 行相邻，为什么不会互相 Attention？**

答案：每行使用所属请求的页表和 context，而不是把整个 packed 张量当成一个共享历史序列。

**题 4：将 context 全填为本轮最大长度能减少分支吗？**

答案：会改变语义，短行可能读未来位置、未初始化槽或不存在的页。不能以这种方式换取规则形状。

**题 5：一个 32 Token Prompt 能 packed 计算，为什么不能一次 Decode 32 个普通 greedy 输出？**

答案：Prompt 的 ID 全已知，可以并行构造当前层输入；后续输出 ID 依赖前一次采样，当前引擎没有实现投机解码等额外机制。

下一篇：[任务 06：存储精度与计算精度](task_06_mixed_precision_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[分页与 Packed 元数据的完整手算](from_pytorch/03_pages_and_packed.md#7-一组元数据完整手算)。
先分清请求数 B、输入行数 N、采样行数 R，再读本任务的矩阵化执行。

**状态：已完成 Packed Prefill 基线。**

## 为什么要做这个任务

任务 04 已经实现完整 GPU Decode，但 Prompt 仍被拆成 T=1 微步。一个长度 32 的 Prompt
会重复执行 32 次完整 12 层 Transformer。线性层只能处理很小的 Batch，容易退化成
GEMV；LayerNorm、Bias、Residual 和 Attention 等 Kernel 也会重复启动。

任务 04 的 Nsight Systems 结果提供了直接证据：一次 Warmup 加一次正式运行包含 86 个
微批次，共启动 17,576 个 Kernel；cuBLAS 类 Kernel 占 GPU Kernel 时间约 63%。因此本轮
先扩大矩阵的 Token 维度，而不是继续只优化占 8.7% 的 PagedAttention。

## Packed ModelInput

旧路径按 `micro_step` 构造多个 ModelInput：

```text
请求 A：a0 a1 a2 a3
请求 B：b0 b1

micro batch 0: [a0, b0]
micro batch 1: [a1, b1]
micro batch 2: [a2]
micro batch 3: [a3]
```

新路径把同一 Scheduler Step 中的 Token 压成一个 Batch：

```text
packed tokens:         [a0, a1, a2, a3, b0, b1]
query_start_locations: [0,              4,      6]
```

每个 Token 仍拥有自己的：

- 绝对 `position`；
- `context_length = position + 1`；
- 新 K/V 写入位置 `slot_mapping`；
- 所属请求的 `block_table`。

当前实现为每个 Packed Token 重复一行 Block Table。这使 Kernel 接口简单、正确性容易
验证；后续可改为 `request_index` 间接索引，减少元数据传输。

## 为什么一次处理多个 Token 仍然满足因果性

Transformer 的同一层按两个 Kernel 阶段执行：

1. 批量计算所有 Packed Token 的 Q/K/V，并将所有新 K/V 写入各自物理 Slot。
2. 每个 Query 根据自己的 `context_length` 遍历 Block Table，只读取位置
   `[0, context_length)` 的 K/V。

例如 a3 的 Context Length 为 4，可以读取 a0..a3；a1 的 Context Length 为 2，只读取
a0..a1。虽然同一 Chunk 后面的 K/V 已经写入缓存，较早 Query 不会访问它们，因此不会
看到未来 Token。每层完成后再进入下一层，和标准 Causal Transformer 的层级依赖一致。

Decode 请求也只是一个长度为 1 的 Packed Span，因此可以和 Prefill Span 放在同一次
Forward 中。测试明确构造了一个 Decode 请求与一个 7 Token Prefill 请求共存的 Step。

## 执行路径变化

旧 GPU Runner：

```text
for micro_step in max_chunk_length:
    prepare active requests
    run all 12 Transformer layers
```

新 GPU Runner：

```text
packed = pack every scheduled token
run all 12 Transformer layers once with B = total_tokens
sample the final token of every completed request span
```

cuBLAS 的矩阵形状从大量 `[1 or small B, hidden]` 转为
`[total_tokens, hidden]`，Prompt 阶段由 GEMV 主导转向更大的 GEMM。

## 正确性验证

模型级测试使用 GPT-2 124M，覆盖：

- 一个 Step 内多个异长 Prompt Span；
- Prefill 与 Decode 同轮混合；
- 16→17 Token 跨页；
- 请求完成后物理页释放和复用；
- Query Start Location 与 Packed Token 数量；
- 全部 50,257 个有效词表 logits；
- GPU Argmax 与 CPU 完整前缀 Greedy Reference。

结果：

```text
max_abs_logit_error=0.000267029
all greedy tokens equal CPU full-prefix reference
memcheck: 0 errors
racecheck: 0 hazards
```

相对任务 04 的 `0.00025177` 略有变化，是 cuBLAS 在更大矩阵形状下选择不同计算路径所致；
误差仍很小，生成 Token 完全一致。

## Token Budget Sweep

固定 GPT-2 124M、RTX 3090 FP32、4 个同时到达请求，Prompt 长度 8/16/24/32，每请求
生成 4 Token。每个配置 Warmup 一次，正式重复 3 次，并与独立 CPU 完整前缀输出对齐。

| Token Budget | 总时间中位数 | 输出吞吐 | TTFT P50/P95 | TPOT P50/P95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 110.936 ms | 144.227 tok/s | 50.048 / 101.192 ms | 1.214 / 1.254 ms |
| 4 | 36.551 ms | 437.740 tok/s | 14.986 / 30.875 ms | 1.426 / 1.450 ms |
| 8 | 23.231 ms | 688.735 tok/s | 8.999 / 18.190 ms | 1.637 / 1.660 ms |
| 16 | 14.608 ms | 1095.307 tok/s | 5.234 / 9.839 ms | 1.696 / 1.722 ms |
| 32 | 9.711 ms | 1647.603 tok/s | 2.744 / 5.161 ms | 1.544 / 1.822 ms |
| 64 | 8.704 ms | 1838.148 tok/s | 2.511 / 4.042 ms | 1.541 / 1.789 ms |

任务 04 的逐 Token GPU 基线为 295.708 tok/s。Token Budget 64 的 Packed Prefill 为
1838.148 tok/s，提升 6.2 倍；中位总时间从 54.107 ms 降至 8.704 ms。TPOT 没有出现
同等幅度变化，因为生成阶段仍然一次只产生一个新 Token，主要受 Decode 路径控制。

Budget 1 的吞吐低于旧 GPU 基线，因为 Budget 1 连不同请求的同位置 Token 也无法组成
Batch。它衡量完全串行 Token 调度；旧基线仍能在每个 micro step 中动态 Batch 多个请求。

原始结果：

- `benchmark/results/gpt2_cuda_packed_budget{1,4,8,16,32,64}_rtx3090.json`
- 同名 CSV 文件
- `benchmark/results/gpt2_cuda_packed_budget_sweep_rtx3090.csv`

复现单个配置：

```bash
make GPU_COMPUTE_CAPABILITY=86 benchmark_gpt2_cuda_serving
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  ./benchmark_gpt2_cuda_serving \
  --repeats 3 --token-budget 64 \
  --json benchmark/results/gpt2_cuda_packed_budget64_rtx3090.json \
  --csv benchmark/results/gpt2_cuda_packed_budget64_rtx3090.csv
```

## Nsight Systems 前后对比

两份 Profile 都包含一次 Warmup 和一次正式运行：

| 指标 | 逐 Token 基线 | Packed Budget 64 | 变化 |
| --- | ---: | ---: | ---: |
| ModelRunner 微批次数 | 86 | 10 | -88.4% |
| Kernel Launch 数 | 17,576 | 2,176 | -87.6% |
| GPU Kernel 总时间 | 101.5 ms | 15.5 ms | -84.7% |
| PagedAttention 总时间 | 8.82 ms | 1.37 ms | -84.5% |

Packed 版本中 cuBLAS 类 Kernel 仍约占 GPU Kernel 时间的 67%，但绝对时间已经大幅下降。
这说明下一阶段使用 FP16/BF16 和 Tensor Core 有清晰依据。PagedAttention 的绝对时间也
因调用次数下降而缩短，不需要修改其数学实现就获得了系统级收益。

Nsight 汇总：

- `benchmark/results/gpt2_cuda_packed_nsys_cuda_gpu_kern_sum.csv`
- `benchmark/results/gpt2_cuda_packed_nsys_cuda_api_sum.csv`

## 下一步

下一阶段实现 FP16/BF16 权重、激活与 KV Cache，并使用 Tensor Core：

1. 保留 FP32 LayerNorm/Softmax 累加，明确混合精度边界。
2. 转换 checkpoint 权重，线性层使用 `cublasGemmEx` 或 cuBLASLt。
3. PagedAttention 支持 half/BF16 Cache 和向量化加载。
4. 比较 logits 误差、生成 Token、显存占用、TTFT、TPOT 和吞吐。
5. 低精度稳定后再做 Bias/Residual/LayerNorm Fusion 与 CUDA Graph。

面试时应重点解释：Packed Token 为什么仍满足因果性；`query_start_locations` 如何描述
异长请求；Token Budget 为什么同时影响 TTFT、吞吐与 TPOT；以及为什么 Nsight 的绝对
时间比单看百分比更能指导优化顺序。
