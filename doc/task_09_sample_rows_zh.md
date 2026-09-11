# 任务 09：只为需要采样的行计算 LM Head

本任务从一个你熟悉的 PyTorch 表达式开始：生成只需要 `logits[:, -1]`。进一步思考，能否在昂贵的词表投影之前就选出需要的 hidden 行？当前实现正是这样做的。

学习目标：区分输入行 N、调度请求数 S、采样行数 R；沿着索引把 packed hidden、紧凑 logits 和请求输出连接起来。

## 1. 优化发生在模型末尾的哪个位置

```text
所有 N 行输入
  → Embedding
  → 所有 Transformer 层（仍计算 N 行并写入 KV）
  → 最终 LayerNorm [N,C]
  → Gather 需要采样的 R 行 [R,C]
  → LM head [R,Vp]
  → Argmax [R]
  → 映射为调度请求顺序 [S]，无输出处填 -1
```

只裁剪 LM head 的输入行，不裁剪前面的 Transformer 行。Prompt 中间位置虽然不对外采样，它们的 KV 仍是后续 Attention 的必要输入。

假设一个请求 Prompt 长 32，预算足够一次完成。主体模型必须计算 32 行，但只需最后一行投影到词表。这是避免无用输出计算，不是跳过 Prompt 理解。

## 2. 先读三个调用点

| 位置 | 建立的映射 |
| --- | --- |
| [run 中建立采样行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L762) | 请求资格 → packed 行号 |
| [Gather kernel](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L638) | packed hidden → 紧凑 hidden |
| [run 中还原输出](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L782) | 紧凑采样数组 → 请求顺序 |
| [独立回归测试](../dev/cuda/test_gpt2_cuda_sample_rows.cu) | 相同 N、不同 R、不同索引的数值一致性 |

你可以在纸上给三个索引起不同名字：`item_index`、`packed_row`、`sample_index`。如果都叫 batch index，阅读时很容易把它们混用。

## 3. 哪些请求具有采样资格

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 762—775 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L762)。以下为当前文件的原样摘录。

```cpp
        last_logit_token_indices_.clear();
        if (config_.enable_sample_row_pruning) {
            for (std::size_t i = 0; i < output.items.size(); ++i) {
                const auto& item = output.items[i];
                if (item.sequence->num_computed_tokens() + item.num_scheduled_tokens ==
                    item.sequence->num_tokens()) {
                    last_logit_token_indices_.push_back(static_cast<int>(
                        input.query_start_locations[i + 1] - 1));
                }
            }
        } else {
            last_logit_token_indices_.resize(input.batch_size());
            std::iota(last_logit_token_indices_.begin(), last_logit_token_indices_.end(), 0);
        }
```

判断条件是 `computed + scheduled == num_tokens`：这轮结束时，所有已知输入都被处理完。条件成立后，才取 `qstart[i+1]−1` 作为采样行。

两个步骤不能反过来简化成“每个请求总有一行可采样”。中间 Prefill chunk 的最后行仍不是整个 Prompt 的最后行。

关闭裁剪时使用 `iota(0..N−1)`，这是可对照的全行投影基线。它产生更多 logits，但最终提交给请求的样本依然只来自具有采样资格的行。

## 4. 先做一次纯 PyTorch 等价推导

设 H 为最终归一化后的 hidden，W 为词表权重，rows 为需要采样的行：

```python
# 教学等价式；不表示运行时会同时计算两份。
full_logits = H @ W.T
expected = full_logits[rows]
actual = H[rows] @ W.T
```

线性投影对行独立，所以“先投影再取行”与“先取行再投影”在数学上相同。浮点实现可能因矩阵尺寸改变而选择不同 kernel，仍需要容差验证。

为什么不把 Gather 移到 Attention 前？Attention 中这些位置承担历史 K/V 的角色，行间有因果依赖；只保留采样行会丢失后续需要的历史表示。

最终 LN 对行独立，理论上可以讨论更早裁剪某些末尾计算，但当前实现的明确边界是**最终 LN 之后、LM head 之前**。阅读与简历描述以这条实际路径为准。

## 5. Gather 的每个线程在搬什么

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 638—647 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L638)。以下为当前文件的原样摘录。

```cpp
// Gather 只选中每个已完成输入请求的最后一行，避免 Prompt 全行 LM Head。
template <typename T>
__global__ void gather_sample_rows_kernel(
    T* output, const T* input, const int* rows, int count, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * channels) return;
    const int row = index / channels;
    const int channel = index % channels;
    output[index] = input[static_cast<std::size_t>(rows[row]) * channels + channel];
}
```

输出按 `[R,C]` 展平。线程全局 index 对应：

```text
compact_row = index / C
channel     = index % C
source_row  = rows[compact_row]
source_idx  = source_row*C + channel
```

教学例子 C=3，输入：

```text
H = [[10,11,12],
     [20,21,22],
     [30,31,32],
     [40,41,42]]
rows = [1,3]
Gather(H) = [[20,21,22], [40,41,42]]
```

线程 index=4 属于紧凑行 1、channel 1，读取原始行 `rows[1]=3` 的第 1 个维度，也就是 41。

这不是对 Token ID 做索引，也不是把 vocab 的某些列删掉。投影后的每个采样行仍保留完整词表，以进行正确的 greedy argmax。

## 6. R=0 时哪些工作省掉，哪些仍必须执行

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1302—1318 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1302)。以下为当前文件的原样摘录。

```cpp
        if (num_logit_rows > 0) {
            const T* lm_input = normalized_.get<T>();
            if (config_.enable_sample_row_pruning) {
                gather_sample_rows_kernel<T><<<
                    blocks_for(num_logit_rows * channels), kThreads, 0, stream_.get()>>>(
                    sampled_hidden_.get<T>(), normalized_.get<T>(), sample_rows_.get(),
                    num_logit_rows, channels);
                check_last_kernel("gather_sample_rows_kernel");
                lm_input = sampled_hidden_.get<T>();
            }
            logits_matmul(lm_input, parameters_view.wte,
                          num_logit_rows, channels, config_.padded_vocab_size);
            argmax_kernel<<<num_logit_rows, kThreads, 0, stream_.get()>>>(
                logits_.get(), sampled_token_ids_.get(), num_logit_rows,
                config_.vocab_size, config_.padded_vocab_size);
            check_last_kernel("argmax_kernel");
        }
```

当本轮只有未完成的 Prefill chunk，R=0，代码跳过 Gather、LM head 和 Argmax。之后不会下载空的采样数组。

但主体模型、KV 写入、调度计数更新仍然必须完成。也仍会在 Runner 返回前同步，保证下一轮读取的历史 KV 已就绪。

因此 R=0 是一个正常且重要的执行形状，不是“空请求”。若错误地在 run 开头直接返回，下一轮 computed 看似前进，实际 KV 却没有写入。

## 7. 紧凑输出如何映射回请求

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 782—795 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L782)。以下为当前文件的原样摘录。

```cpp
        std::size_t sample_index = 0;
        for (std::size_t item_index = 0;
             item_index < output.items.size(); ++item_index) {
            const ScheduledItem& item = output.items[item_index];
            const Sequence& sequence = *item.sequence;
            if (sequence.num_computed_tokens() +
                    item.num_scheduled_tokens ==
                sequence.num_tokens()) {
                const std::size_t final_token =
                    input.query_start_locations[item_index + 1] - 1;
                sampled[item_index] = token_samples[
                    config_.enable_sample_row_pruning ? sample_index++ : final_token];
            }
        }
```

`sample_index` 只在具有采样资格的 item 上递增。假设三个请求 A/B/C，只有 A 和 C 完成输入：

```text
请求顺序：       [A, B, C]
采样行号：       [A_last_row, C_last_row]
GPU token_samples：[token_A, token_C]
返回 sampled：   [token_A, -1, token_C]
```

若用 `token_samples[item_index]`，C 会访问第 2 个下标，而紧凑数组只有两个元素；即使某次没有越界，也可能把别人的 Token 交给错误请求。

这段映射与创建 rows 时采用同样的资格条件和请求遍历顺序，是两端能够对齐的原因。修改其中一端时必须同步检查另一端。

## 8. 用测试里的五轮输入推导 Graph 行为

源码：[dev/cuda/test_gpt2_cuda_sample_rows.cu，第 23—24 行](../dev/cuda/test_gpt2_cuda_sample_rows.cu#L23)。以下为当前文件的原样摘录。

```cpp
        const std::vector<std::vector<int>> lengths{{8}, {4}, {2, 2}, {2, 8}, {8}};
        const std::vector<std::vector<int>> expected_rows{{}, {3}, {1, 3}, {1}, {}};
```

测试始终调度 N=4 行，单请求时调度 4 个 Token，两请求时各调度 2 个：

| 测试 | 请求 Prompt 长度 | 本轮长度 | 需要采样的 packed 行 | R |
| --- | --- | --- | --- | ---: |
| 0 | `[8]` | `[4]` | `[]` | 0 |
| 1 | `[4]` | `[4]` | `[3]` | 1 |
| 2 | `[2,2]` | `[2,2]` | `[1,3]` | 2 |
| 3 | `[2,8]` | `[2,2]` | `[1]` | 1 |
| 4 | `[8]` | `[4]` | `[]` | 0 |

第 1 与第 3 个测试都使用 `(N=4,R=1)`，但采样行从 3 变为 1。这正好验证同一张 Graph replay 时会读取新的 sample_rows 内容，而不是保留第一次的行号。

最终图数量应为 3，按 `(4,0)`、`(4,1)`、`(4,2)` 缓存，不能按运行次数增长到 5，也不能只按 N 保留 1 张。

## 9. 数值检查为什么需要索引对齐

源码：[dev/cuda/test_gpt2_cuda_sample_rows.cu，第 41—58 行](../dev/cuda/test_gpt2_cuda_sample_rows.cu#L41)。以下为当前文件的原样摘录。

```cpp
            const auto samples_a = pruned.run(a), samples_b = full.run(b);
            assert(samples_a == samples_b);
            assert(pruned.last_logit_token_indices() == expected_rows[test]);
            assert(pruned.num_cuda_graphs() == std::min(test + 1, std::size_t{3}));
            const auto la = pruned.last_logits_for_testing(), lb = full.last_logits_for_testing();
            assert(la.size() == expected_rows[test].size() * model.config.padded_vocab_size);
            assert(lb.size() == 4ul * model.config.padded_vocab_size);
            for (std::size_t row = 0; row < expected_rows[test].size(); ++row) {
                for (int v = 0; v < model.config.vocab_size; ++v) {
                    const double error = std::abs(double(la[row * model.config.padded_vocab_size + v]) -
                        lb[expected_rows[test][row] * model.config.padded_vocab_size + v]);
                    max_error = std::max(max_error, error);
                }
            }
            for (auto& item : a.items) pruned_blocks.release(*item.sequence);
            for (auto& item : b.items) full_blocks.release(*item.sequence);
        }
        assert(max_error < 0.005);
```

紧凑 logits 的第 row 行，对照全量 logits 的 `expected_rows[row]` 行。直接比较两块数组前 R 行会拿错参考位置。

测试同时检查样本、行映射、图数量、缓冲元素数以及真实词表上的 logits 误差。它没有因为 greedy 一致就省略所有数值检查。

对 `R=0`，调试 logits 应是空向量；全行基线仍有 N 行。这能发现“跳过计算但误返回上一轮 logits”的状态残留问题。

当前阈值 `0.005` 是该测试对同精度两条执行路径的要求，不等于所有模型/精度普遍适用的容差标准。

## 10. 显存与计算量怎样计算

未裁剪 logits 容量是 `max_num_tokens*Vp*4` 字节；裁剪后最多每请求一行，因此是 `max_num_sequences*Vp*4` 字节，另增加较小的 `[max_sequences,C]` Gather 缓冲。

教学取 Vp=50304、max_tokens=64、max_sequences=4：

```text
旧 logits：64*50304*4 = 12,877,824 字节 = 12.28125 MiB
新 logits： 4*50304*4 =    804,864 字节 =  0.767578125 MiB
```

这是 logits 单项容量，不是整个 Runner 激活总量。文末实测 `13.781→2.180 MiB` 是按该实验配置统计的全部激活缓冲，不能把两组数字混为一谈。

同一工作负载四个 Prompt 共 80 Token，各生成 4 个：全量路径 LM head 总行数为 92，裁剪后为 16。主体 Transformer 仍处理 92 行，不能宣称整个模型计算量都减少到 16/92。

## 11. 为什么总时间只小幅下降

裁剪效果取决于 LM head 在总时间中的占比。即使某一部分减少很多，其他 Transformer 层、KV Attention、元数据和同步仍存在；新增 Gather 也有成本。

仓库固定短负载已有结果只显示小幅时间改善，见文末原始数据。主要可以明确陈述的是“减少无用词表投影行与 logits 缓冲”，不要把显存下降比例写成端到端加速比例。

复现时全行基线要显式传 `--full-logits`，否则当前默认已经启用裁剪，两次运行可能实际走同一条路径。

## 12. 练习与答案

**题 1：三个请求本轮长度 `[2,3,1]`，只有前两个处理完全部输入，rows 是什么？**

答案：qstart=`[0,2,5,6]`，rows=`[1,4]`，R=2。第三个 chunk 的最后行 5 不可采样。

**题 2：R≤S≤N 总成立吗？**

答案：对当前非空调度、每 item 至少一个 Token、每请求最多一个采样的路径成立。不要把这个关系无条件推广到束搜索或其他一次多候选输出接口。

**题 3：R=0 是否可以跳过所有 CUDA 调用？**

答案：不能，必须计算本轮 hidden 与 KV，只跳过末尾采样所需的投影链路。

**题 4：从全词表 logits 中只保留最高分的几个 vocab 列，是同一个优化吗？**

答案：不是。本任务选的是输入 Token 行，保留每行完整词表；限制词表列会涉及另一种语义和实现。

**题 5：为什么调试时要同时读取 last_logit_token_indices？**

答案：logits 行已经紧凑排列，需要这个索引才能映射到原 packed 行的请求和 position。

下一篇：[任务 10：把 Prefix Cache 的收益测清楚](task_10_prefix_benchmark_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[生成与 KV](from_pytorch/01_generation_and_kv.md)、[Packed 元数据](from_pytorch/03_pages_and_packed.md)。
先运行 CPU packed 实验，理解为何只需每个完成输入请求的末行，再看 CUDA Gather。

本任务只改模型末尾的词表投影和 Argmax。先学完任务 05 的 Packed Prefill，再读本篇。
默认开启 `enable_sample_row_pruning`；`--full-logits` 可以恢复全行投影，便于验证和对照。

## 1. 学习顺序与代码地图

| 顺序 | 要理解的内容 | 实现位置 | 谁调用它 |
| --- | --- | --- | --- |
| 1 | 哪个请求需要输出 Token | [Runner::run](../mini_vllm/cuda/gpt2_cuda_model_runner.cu)，搜索 `last_logit_token_indices_.clear` | `GPT2CudaEngine::step` 或 `GPT2PDEngine::step` |
| 2 | 请求最后一行在哪里 | [prepare_packed_model_input](../mini_vllm/model_input.hpp)，搜索 `query_start_locations` | Runner 在 `run` 开头调用 |
| 3 | 把少数行搬到连续矩阵 | [gather_sample_rows_kernel](../mini_vllm/cuda/gpt2_cuda_model_runner.cu) | `forward<T>` 在最终 LayerNorm 后调用 |
| 4 | 词表 GEMM 行数缩小 | 同文件 `logits_matmul` 和 `num_logit_rows > 0` | Gather 后调用 |
| 5 | Graph 如何区分形状 | 同文件 `graph_key`、`cuda_graphs_` | `forward<T>` |
| 6 | 返回值如何还原到请求 | 同文件 `sample_index` | Runner::run 返回前 |
| 7 | 独立 A/B 验证 | [test_gpt2_cuda_sample_rows.cu](../dev/cuda/test_gpt2_cuda_sample_rows.cu) | 测试程序 `main` |
| 8 | 服务性能对照 | [benchmark_gpt2_cuda_serving.cu](../benchmark/benchmark_gpt2_cuda_serving.cu) | `run_once`、`write_json` |

代码行号会随着后续任务变化，表中搜索词是稳定定位点。可以使用：

```bash
rg -n 'last_logit_token_indices_|gather_sample_rows_kernel|graph_key|sample_index' \
  mini_vllm/cuda/gpt2_cuda_model_runner.cu
```

## 2. 为什么可以减少行数

GPT-2 末尾先得到隐藏状态，再乘词嵌入权重，生成词表 logits：

```text
原来：normalized [N, C] × Wteᵀ [C, V] → logits [N, V]
现在：Gather → sampled_hidden [R, C] × Wteᵀ [C, V] → logits [R, V]
```

`N` 是本轮输入 Token 总数，`R` 是本轮完成输入、需要采样的请求数。
以 64 Token Prompt 的单请求 Prefill 为例，下一 Token 来自最后一个位置的 logits，前 63 行
词表分数并不会被使用。所有 64 个位置的 Transformer 与 KV 写入仍然执行，否则后续
Attention 缺少历史上下文。

Decode 时每个请求通常只输入一个 Token，因此 `N = R`，这一轮不再减少词表行数，反而多
一个很小的 Gather。最终是否加速应看完整工作负载，不能从行数减少直接推断整个引擎快几倍。

## 3. 从调度结果确定采样行

入口在 `GPT2CudaEngine::step`：

```cpp
SchedulerOutput output = scheduler_.schedule();
result.sampled_token_ids = model_runner_.run(output);
scheduler_.commit(output, result.sampled_token_ids);
```

进入 `Runner::run` 后先构造 Packed 输入，再判断每个请求：

```cpp
if (item.sequence->num_computed_tokens() + item.num_scheduled_tokens ==
    item.sequence->num_tokens()) {
    last_logit_token_indices_.push_back(static_cast<int>(
        input.query_start_locations[i + 1] - 1));
}
```

左边表示“本轮执行完后已计算到哪里”，右边表示“当前实际拥有多少 Token”。两者相等，才有
资格生成下一个 Token。`query_start_locations[i + 1] - 1` 是该请求在 Packed Batch 的最后一行，
它与请求自己的绝对 Position 不是同一个数字。

例子：A 本轮计算 3 个 Token 并完成 Prompt；B 本轮计算 5 个 Token，但 Prompt 还剩 2 个：

```text
Packed row:       0  1  2 | 3  4  5  6  7
Request:          A  A  A | B  B  B  B  B
query_start:      [0, 3, 8]
sample_rows:      [2]
N = 8，R = 1
```

只对行 2 做词表投影。返回值仍按请求排列，为 `[A 的 Token, -1]`，Scheduler 原有接口不变。

## 4. Gather 与 LM Head 的调用点

`forward<T>` 在图外上传 `sample_rows`，随后在计算链路末尾调用：

```cpp
if (num_logit_rows > 0) {
    // enable_sample_row_pruning 分支：从 normalized 选出需要的行。
    gather_sample_rows_kernel<T><<<..., kThreads, 0, stream_.get()>>>(
        sampled_hidden_.get<T>(), normalized_.get<T>(), sample_rows_.get(),
        num_logit_rows, config_.channels);
    // 随后的 logits_matmul 和 Argmax 都使用 num_logit_rows。
}
```

上面省略了未裁剪分支和 Launch 网格的算式。完整代码搜索 `const T* lm_input`。
Gather Kernel 的实际索引计算是：

```cpp
const int row = index / channels;
const int channel = index % channels;
output[index] = input[static_cast<std::size_t>(rows[row]) * channels + channel];
```

这个 Kernel 只做数据复制，保持 FP16/FP32 存储格式。输入仍有 N 行，输出连续放置 R 行，方便
cuBLAS 按常规矩阵布局计算。FP16 GEMM 的输出 logits 仍为 FP32。

整个 Chunk 都未完成任何请求时 `R = 0`：Transformer 正常执行、KV 正常写入；跳过 Gather、
LM Head、Argmax 和样本 D2H。Stream 仍同步完成，Scheduler 才能安全更新已计算 Token 数。

## 5. CUDA Graph 的缓存键必须跟着改变

当前实现：

```cpp
const auto graph_key = std::make_pair(batch_size, num_logit_rows);
const auto graph = cuda_graphs_.find(graph_key);
```

`N` 控制 Transformer Kernel 网格和 GEMM 形状，`R` 控制 LM Head 和 Argmax 形状。对于同一个
N=4 的 Batch，R 可能是 0、1、2，不能使用同一张捕获图。

具体行索引不加入 Key，因为它是图外更新的设备数组内容。例如 `(N=4,R=1)` 第一次采样行 3，
下一次采样行 1，Replay 仍从固定 `sample_rows_` 地址读取新的值。

学习时可从三个角度检查：

- 固定地址：DeviceBuffer 在 Runner 构造时分配，Replay 期间不重新分配。
- 固定形状：同一个 Graph Key 保证 N、R 和对应 Launch 参数相同。
- 动态数据：Token、Position、页表和采样行索引在同一 Stream、Graph Launch 之前更新。

## 6. 返回值和调试 logits 的变化

Runner 最后的映射代码：

```cpp
sampled[item_index] = token_samples[
    config_.enable_sample_row_pruning ? sample_index++ : final_token];
```

`sample_index` 只在需要采样的请求上递增。不能直接用 `item_index` 索引压缩后的输出，因为
中间可能夹着未完成的 Chunk。

`last_logits_for_testing()` 现在返回 `R × padded_vocab_size`，而不是总输入 Token 数乘词表。
调用 `last_logit_token_indices()` 就能知道其中每行对应哪一行 Packed 输入。`--full-logits`
下返回 N 行，映射为 `[0,1,...,N-1]`。

构造中的 `max_logit_rows()` 也随开关变化：裁剪路径用最大请求数，完整路径用最大 Token 数。
因此除了计算量，也减少持久 logits 缓冲区；其他中间激活仍按最大输入 Token 数分配。

## 7. 验证与复现

```bash
conda activate zyf1
make test_gpt2_cuda_sample_rows benchmark_gpt2_cuda_serving GPU_COMPUTE_CAPABILITY=86
OMP_NUM_THREADS=8 ./test_gpt2_cuda_sample_rows
OMP_NUM_THREADS=8 ./test_gpt2_cuda_sample_rows --fp32
```

测试固定 N=4，让 R 依次为 `0,1,2,1,0`。它同时检查：

- 裁剪 + Graph 与完整 logits + Eager 的 Greedy Token 一致。
- 被选中的整行 50,257 个有效词表值误差小于 0.005。
- 第四个 Case 改变采样行位置，Graph 数仍保持 3，证明是更新数据后复用。
- 第五个 Case 重放无采样图，调试 logits 返回空数组。

性能复现见 [任务 09—11 结果](../benchmark/results/task09_11/README.md)。比较时使用相同精度、
Fusion 开关、Token Budget、请求数据和重复次数，只改变 `--full-logits`，Graph 再单独成组。

## 8. 学完后自己做的练习

1. 手算 Prompt 长度 `[3,9]`、每个请求本轮执行 `[3,5]` 时的 `query_start_locations` 和采样行。
2. 将测试第四个 Case 改成另一种需要采样的位置，解释为什么 Graph Key 不增加。
3. 根据 JSON 中 `projected_rows` 解释：处理 80 个 Prompt Token、生成 16 个新 Token，为何全行
   路径投影 92 行，裁剪路径投影 16 行，而不是投影 80 行和 4 行。
4. 在 Nsight 里定位 Gather → GEMM → Argmax；找一个未完成 Chunk，确认它只执行模型主干。
