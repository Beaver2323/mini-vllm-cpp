# 任务 09：只为需要采样的行计算 LM Head

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
