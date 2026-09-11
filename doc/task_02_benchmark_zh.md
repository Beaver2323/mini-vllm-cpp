# 开发任务 02：建立可复现的推理 Benchmark

本任务教你把“我感觉变快了”变成可以审查的实验。你不需要先懂 vLLM 的 Benchmark 工具；先跟着本项目的 CPU 脚本，把请求、计时点、输出数和统计口径对齐。

学习目标：能修改一个工作负载、解释三种模式各自比较了什么，并从原始时间重新算出 TTFT、TPOT 与整批吞吐。

## 1. 阅读地图：测量代码也是需要读懂的代码

| 阅读顺序 | 源码位置 | 核心问题 |
| --- | --- | --- |
| 1 | [工作负载](../benchmark/benchmark_gpt2_serving.cpp#L105) | 输入是否一致、什么时候到达？ |
| 2 | [完整重算路径](../benchmark/benchmark_gpt2_serving.cpp#L170) | 是否偷偷多做或少做了工作？ |
| 3 | [分页引擎路径](../benchmark/benchmark_gpt2_serving.cpp#L220) | 分配与提交是否算入时间？ |
| 4 | [统计函数](../benchmark/benchmark_gpt2_serving.cpp#L124) | P95 究竟怎样得到？ |
| 5 | [输出一致性](../benchmark/benchmark_gpt2_serving.cpp#L310) | 比较速度前如何证明输出可比？ |
| 6 | [主程序与 warmup](../benchmark/benchmark_gpt2_serving.cpp#L600) | 预热和重复发生在哪里？ |

```text
main
  ├─ 加载模型 / make_workload
  ├─ 每种模式 warmup 一轮并核对输出
  ├─ repeats 次正式运行
  │   ├─ run_full_recompute
  │   ├─ run_paged_engine(max_num_sequences=1)
  │   ├─ run_paged_engine(max_num_sequences=4)
  │   └─ verify_same_outputs
  └─ summarize_mode → CSV / JSON
```

`run_paged_engine` 的两个调用复用相同实现，通过最大并发请求数区分模式。它们不是两个不同 Attention kernel，不能把两者差异归因于算子代码改变。

## 2. 输入生成：长度相同还不够，Token 也要相同

源码：[benchmark/benchmark_gpt2_serving.cpp，第 105—117 行](../benchmark/benchmark_gpt2_serving.cpp#L105)。以下为当前文件的原样摘录。

```cpp
static std::vector<WorkloadRequest> make_workload(
    std::size_t max_new_tokens) {
    const std::vector<std::size_t> prompt_lengths = {8, 16, 24, 32};
    std::vector<WorkloadRequest> workload;
    for (std::size_t i = 0; i < prompt_lengths.size(); ++i) {
        workload.push_back({
            i + 1,
            make_prompt(prompt_lengths[i], checked_int(i + 1, "seed overflow")),
            max_new_tokens,
        });
    }
    return workload;
}
```

四个 Prompt 长度为 8、16、24、32，`max_new_tokens` 统一从参数传入。请求 ID 用于关联指标，不是词表 ID，也不是物理页 ID。

这里用固定生成规则构造 Token 序列，便于重复实验。它测的是模型执行与调度，不包含聊天模板、文本分词和网络传输。因此简历中的性能结论应限定为“固定 Token 工作负载”，不能写成线上聊天服务吞吐。

全部请求在同一计时起点已到达。即使某个请求还在 waiting，它的 TTFT 也在增长。把计时起点移到“请求终于获准执行”会隐去排队成本，使串行模式看起来不合理地好。

可以先做一个预算手算：默认四个请求各输出 4 个 Token，合计输出 16 个；输入处理量与输出量是不同的分母。

| 模式 | 实际输入 Token 次数（不计批处理组织差异） |
| --- | ---: |
| 完整重算 | `(8+9+10+11)+(16+17+18+19)+(24+25+26+27)+(32+33+34+35)=344` |
| 使用 KV | `(8+3)+(16+3)+(24+3)+(32+3)=92` |

CPU KV 路径少处理了历史 Token，但仍可能更慢，因为矩阵形状、访存和调度开销也决定执行效率。这正是需要测量而不能只数 FLOPs 的原因。

## 3. 基线前向：哪一行 logits 用来产生输出

源码：[benchmark/benchmark_gpt2_serving.cpp，第 170—198 行](../benchmark/benchmark_gpt2_serving.cpp#L170)。以下为当前文件的原样摘录。

```cpp
static RunMetrics run_full_recompute(
    const GPT2& model, const std::vector<WorkloadRequest>& workload,
    std::size_t max_context_length, int repeat) {
    GPT2DenseInferenceWorkspace workspace(
        model.config, /*max_batch_size=*/1,
        checked_int(max_context_length, "context is too large"));
    std::vector<RequestMetrics> request_metrics;
    std::vector<std::vector<int>> outputs;
    std::size_t output_token_count = 0;

    const Clock::time_point benchmark_start = Clock::now();
    for (const WorkloadRequest& request : workload) {
        std::vector<int> tokens = request.prompt_tokens;
        std::vector<int> generated;
        double first_token_ms = 0.0;
        for (std::size_t step = 0; step < request.max_new_tokens; ++step) {
            gpt2_forward_dense_with_workspace(
                &model, tokens.data(), 1, checked_int(
                    tokens.size(), "sequence is too long"), &workspace);
            const float* last_logits =
                workspace.acts().logits +
                (tokens.size() - 1) * model.config.padded_vocab_size;
            const int next_token =
                greedy_argmax(last_logits, model.config.vocab_size);
            tokens.push_back(next_token);
            generated.push_back(next_token);
            const double now_ms =
                elapsed_ms(benchmark_start, Clock::now());
            if (step == 0) first_token_ms = now_ms;
```

逐段解释：

1. Workspace 在 `benchmark_start` 之前创建，避免把反复申请内存当成“完整重算必然有的开销”。
2. 外层遍历请求意味着这个基线是串行请求执行。
3. 内层每次把整个 `tokens` 送入模型，所以历史 Token 会反复计算。
4. `last_logits` 跳过前 `tokens.size()-1` 行，取最后一行。
5. 计算完成后才追加新 ID。因此下一次循环的输入比上一次多 1。
6. `first_token_ms` 相对整批起点记录，包含前面请求占用的时间。

PyTorch 语义可以写为：

```python
# 教学对照；C++ 使用预分配的 Workspace。
for request in requests:
    ids = list(request.prompt)
    for j in range(request.max_new_tokens):
        logits = model(torch.tensor(ids)[None, :])
        next_id = logits[0, -1, :vocab_size].argmax().item()
        ids.append(next_id)
```

注意 Python 的 `.item()` 在 CUDA Tensor 上通常引入等待；这里对应的是 CPU 实现。将来比较 GPU 时不能直接删掉等待又沿用相同计时解释。

## 4. 分页模式：预留容量与峰值使用量是两回事

源码：[benchmark/benchmark_gpt2_serving.cpp，第 224—242 行](../benchmark/benchmark_gpt2_serving.cpp#L224)。以下为当前文件的原样摘录。

```cpp
    std::size_t total_blocks = 0;
    for (const WorkloadRequest& request : workload) {
        const std::size_t max_processed_tokens =
            request.prompt_tokens.size() + request.max_new_tokens - 1;
        total_blocks +=
            (max_processed_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
    }
    GPT2Engine engine(
        model, total_blocks,
        {max_num_sequences, /*max_num_batched_tokens=*/64},
        max_context_length);

    std::vector<std::shared_ptr<Sequence>> sequences;
    for (const WorkloadRequest& request : workload) {
        sequences.push_back(engine.add_request(
            request.request_id, request.prompt_tokens,
            SamplingParams{request.max_new_tokens, -1, false}));
    }
    std::vector<Clock::time_point> first_token_times(workload.size());
```

容量按每个请求最大处理长度 `prompt + output − 1` 算页数再求和，确保这个基准不会因为页不足而改变实验主题。

`total_blocks` 是池容量，不是实际峰值。`max_num_sequences=1` 时，前一个请求释放的页可供后一个请求复用，实际同时使用的页通常少于池容量。

页大小 16，默认四个请求最多分别处理 11、19、27、35 个 Token，各需要 1、2、2、3 页，容量合计 8 页。这个值与“分配了 8 个连续请求缓冲”含义不同，页池允许请求按需领取物理页。

输入入队也位于正式计时之前。后面的 Prefix Benchmark 和 PD Benchmark 计时包含 `add_request`，阅读跨任务结果时要核对计时边界，不能仅凭指标列名相同直接横比。

## 5. 计时观察点：为什么是 step 返回之后

源码：[benchmark/benchmark_gpt2_serving.cpp，第 248—272 行](../benchmark/benchmark_gpt2_serving.cpp#L248)。以下为当前文件的原样摘录。

```cpp
    const Clock::time_point benchmark_start = Clock::now();
    while (!engine.is_finished()) {
        std::vector<std::size_t> completion_counts;
        for (const auto& sequence : sequences) {
            completion_counts.push_back(
                sequence->num_completion_tokens());
        }
        const EngineStepResult step = engine.step();
        peak_blocks = std::max(
            peak_blocks,
            engine.num_blocks() - step.free_blocks_after_schedule);
        const Clock::time_point now = Clock::now();
        for (std::size_t i = 0; i < sequences.size(); ++i) {
            if (!saw_first_token[i] &&
                sequences[i]->num_completion_tokens() >
                    completion_counts[i]) {
                first_token_times[i] = now;
                saw_first_token[i] = true;
            }
            if (!saw_completion[i] && sequences[i]->is_finished()) {
                completion_times[i] = now;
                saw_completion[i] = true;
            }
        }
    }
```

循环开始保存每个请求的输出数；执行后用“输出数是否增加”判断首 Token 出现。这样不会把 Prefill 的一个中间 chunk 错当成首 Token。

本轮所有请求使用同一个 `now`，因为对外可观察的边界是 `engine.step()` 返回。CPU Runner 可能早在某个微步算出了 Decode 的采样结果，但 Scheduler 要等本轮微步都结束才提交。

因此长 Prefill 可能推迟同轮 Decode 输出的可见时间。理解这个观察点，才能解释为何吞吐提高同时 TPOT 可能恶化。

`free_blocks_after_schedule` 用于统计峰值，而不是 `free_blocks_after_commit`。后者可能已经归还完成请求的页，会低估本轮模型执行期间真正占用的资源。

## 6. 从时间轴推导指标

假设所有请求在 `t=0` 到达，一个请求输出 4 个 Token，可见时间为：

```text
到达     首 Token      第 2 个       第 3 个      第 4 个 / 完成
0 ms      10 ms          14 ms         20 ms        22 ms
            └── 4 ms ────┴── 6 ms ─────┴── 2 ms ───┘
```

- TTFT = `10−0 = 10 ms`。
- 三个 ITL 分别为 4、6、2 ms。
- TPOT = `(22−10)/(4−1) = 4 ms`，是首 Token 后的平均间隔。
- 请求 latency = `22−0 = 22 ms`。
- 若这是唯一请求，其生成吞吐 = `4 / 0.022 ≈ 181.82 tok/s`。

不能用 `latency/4` 代替 TPOT；那会把 Prefill 和排队摊进去。也不能用 `1/TPOT` 代替多请求整批吞吐，因为不同请求的间隔可能重叠。

源码：[benchmark/benchmark_gpt2_serving.cpp，第 281—303 行](../benchmark/benchmark_gpt2_serving.cpp#L281)。以下为当前文件的原样摘录。

```cpp
                "engine did not record complete request timings");
        }
        const double ttft_ms =
            elapsed_ms(benchmark_start, first_token_times[i]);
        const double completion_ms =
            elapsed_ms(benchmark_start, completion_times[i]);
        const double tpot_ms =
            workload[i].max_new_tokens > 1
                ? elapsed_ms(first_token_times[i], completion_times[i]) /
                      static_cast<double>(
                          workload[i].max_new_tokens - 1)
                : 0.0;
        request_metrics.push_back({
            workload[i].request_id, ttft_ms, tpot_ms, completion_ms});
        const Sequence& sequence = *sequences[i];
        outputs.emplace_back(
            sequence.token_ids().begin() +
                static_cast<std::ptrdiff_t>(
                    sequence.num_prompt_tokens()),
            sequence.token_ids().end());
        output_token_count += outputs.back().size();
    }

```

当输出数为 1 时，首 Token 之后没有间隔，代码约定 TPOT 为 0。这是边界约定，不表示 Decode 无限快，也不适合与多 Token 请求直接比较 TPOT。

## 7. P50/P95：小样本中的插值

源码：[benchmark/benchmark_gpt2_serving.cpp，第 124—132 行](../benchmark/benchmark_gpt2_serving.cpp#L124)。以下为当前文件的原样摘录。

```cpp
static double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}
```

函数先排序，再使用 `fraction*(n−1)` 的小数索引，在相邻样本之间插值。用 `[10,20,30,40]` 手算：

| 指标 | 索引 | 插值 | 结果 |
| --- | ---: | --- | ---: |
| P50 | 1.5 | `20*0.5 + 30*0.5` | 25 |
| P95 | 2.85 | `30*0.15 + 40*0.85` | 38.5 |

所以 P95 不一定是实际出现过的单个请求时间。四个请求算出的 P95 主要方便固定负载比较，不能代表大规模线上尾延迟分布。

还有两层统计要区分：每一轮先算四个请求的 TTFT P50/P95，再跨重复实验取对应指标的中位数。它不等价于把所有轮的所有请求混成一个数组再算 P95。

读取 JSON 时优先核对每轮原始点；只有中位数、没有原始点，无法判断一次异常慢运行是否影响结论。

## 8. 输出一致性是计时结论的前提

源码：[benchmark/benchmark_gpt2_serving.cpp，第 310—317 行](../benchmark/benchmark_gpt2_serving.cpp#L310)。以下为当前文件的原样摘录。

```cpp

static void verify_same_outputs(
    const RunMetrics& expected, const RunMetrics& actual) {
    if (expected.outputs != actual.outputs) {
        throw std::runtime_error(
            "benchmark modes produced different greedy tokens");
    }
}
```

这里比较每个请求的完整 greedy 输出序列。假如其中一种模式提前结束，少生成了一半 Token，却只比较总时间，就会把错误当作优化。

一致的 Token 是必要检查，但不是所有数值正确性的充分证明：两组 logits 可能有误差但 argmax 相同。任务 01、03、04 的独立 reference 测试负责更细的数值检查；Benchmark 的职责是确保当前比较没有明显改变工作内容。

记录一次实验至少应能回答：

| 信息 | 为什么需要 |
| --- | --- |
| commit、编译器和编译参数 | 确定运行的是哪一版实现 |
| 模型、输入 ID、输出 ID | 固定问题本身并证明可比 |
| 线程数、设备和精度 | 确定执行资源与数值路径 |
| token budget、最大并发 | 确定调度策略 |
| warmup、重复数、原始点 | 判断预热和波动 |
| 计时包含哪些环节 | 防止比较不同口径 |

## 9. 你可以怎样做一次有效的小实验

先按文末命令复现，再只改变 OpenMP 线程数，保持模型、请求与输出数不变。把新 JSON 写到新文件，避免覆盖仓库已有证据。

```bash
# 在仓库目录，先完成文末的构建。
OMP_NUM_THREADS=8 conda run -p /home/miniconda3/envs/zyf1 \
  ./benchmark_gpt2_serving --repeats 3 --max-new-tokens 4 \
  --json /tmp/zyf_cpu_omp8.json --csv /tmp/zyf_cpu_omp8.csv
```

先预测：更多线程不一定总是更快，微小 GEMV 的线程协作成本可能抵消收益。运行后检查输出一致性，再比较三种模式各自对线程数的敏感度。

如果分页单请求慢，不要立刻修改 Attention。按这个顺序找证据：

1. 先确认输出、工作量和计时边界。
2. 看 Prefill 在 CPU Runner 中被拆成了多少微步。
3. 区分 Prompt 阶段与生成阶段的占比。
4. 再判断应该改变批处理组织、矩阵形状，还是优化单个 kernel。

后续 GPU 实验还要考虑 CUDA 异步执行。仅围住 kernel launch 的 CPU 时间测到的是提交耗时；本项目 Runner 在返回采样 ID 前同步，因此整轮主机计时包含执行等待。

## 10. 练习与答案

**题 1：整批输出 16 个 Token，用时 8 ms，吞吐是多少？**

答案：`16/(8/1000)=2000 tok/s`。毫秒必须先换成秒；分母是整批墙钟时间，不能把四个请求 latency 相加。

**题 2：请求等待 30 ms，执行 Prefill 5 ms，首 Token 的 TTFT 是多少？**

答案：从到达开始计算时为 35 ms。从执行开始计时得到的 5 ms 只是服务时间的一部分。

**题 3：两轮 TTFT P95 分别为 20 和 40 ms，把所有请求混合后的 P95 必为 30 ms 吗？**

答案：不是。跨轮汇总与跨请求分位数是不同操作，分位数一般不能这样交换计算顺序。

**题 4：KV 模式输入量 92，完整重算 344，可以宣称加速 3.74 倍吗？**

答案：只能说本负载的输入 Token 次数约少到原来的 26.7%。硬件利用率、矩阵尺寸、内存访问与调度开销使执行时间不按这个比例缩放。

**题 5：某优化总时间降低，TPOT 却上升，应否隐藏 TPOT？**

答案：应同时报告。这可能是吞吐与交互延迟的取舍，正是调度优化需要解释的行为。

下一篇：[任务 03：把分页地址真正用于 CUDA Attention](task_03_cuda_paged_attention_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[实验手册中的指标时间线](from_pytorch/06_labs_and_answers.md#9-性能指标也从时间线推导)。
先区分 TTFT、ITL、TPOT 与整批吞吐，再比较本任务的历史数据。

**状态：已完成 CPU 基线。**

## 工作负载

- 模型：GPT-2 124M
- CPU：Intel Xeon Gold 5119T
- OpenMP：16 线程
- 请求数：4，全部在计时起点提交
- Prompt 长度：8、16、24、32
- 每请求输出：4 Token
- KV Block Size：16
- Scheduler Token Budget：64
- 每种模式先 warmup 1 次，再正式运行 3 次
- 三种模式使用相同权重、Prompt Token 和 greedy 输出

## 对比模式

1. `full_recompute`：每生成一个 Token 都重新执行完整前缀前向。
2. `paged_sequential`：使用分页 KV Cache，但每次只运行一个请求。
3. `continuous_batching`：最多同时运行 4 个请求，允许 Decode 与 Chunked Prefill 混合。

模型加载、Workspace/KV Cache 分配和 warmup 均不计入正式时间。所有请求在计时开始前
进入 waiting 队列，因此后加入执行的请求 TTFT 和 latency 包含排队时间。

## 中位数结果

| 模式 | 总时间 | 输出吞吐 | TTFT P50/P95 | TPOT P50/P95 | 请求延迟 P50/P95 | 峰值 Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整前缀重算 | 2.212 s | 7.234 tok/s | 733.0 / 1507.9 ms | 134.4 / 195.4 ms | 1136.3 / 2094.2 ms | 0 |
| 分页单请求 | 4.127 s | 3.877 tok/s | 1822.6 / 3761.7 ms | 45.0 / 45.2 ms | 1958.0 / 3895.0 ms | 3 |
| 连续批处理 | 2.074 s | 7.714 tok/s | 1200.8 / 1812.5 ms | 276.3 / 276.3 ms | 2029.8 / 2067.5 ms | 8 |

在该固定工作负载中，Continuous Batching 的输出吞吐是分页单请求的 1.99 倍。
它与完整前缀重算的吞吐接近，不能据此宣称 PagedAttention 在 CPU 上带来普遍加速。

## 怎样理解结果

CPU 完整前缀重算虽然重复计算历史 Token，但能把多个 Token 合并成较大的矩阵乘；
当前分页增量路径把 Prefill 拆成单 Token 微步，主要执行 GEMV，CPU 利用率较低。

分页单请求在进入 Decode 后 TPOT 最低，但四个请求串行执行导致 TTFT 和总吞吐较差。
Continuous Batching 通过动态 Batch 提高了吞吐和尾部完成时间；当前 Token Budget 为 64，
一次较大的 Prefill chunk 会延迟整轮 commit，因此活跃 Decode 请求的 TPOT 偏高。

这些结果当时给出两个优化方向，后续任务均已完成：

1. Packed Multi-Token Prefill 已将同轮 Token 合并为 GEMM，避免逐 Token GEMV。
2. CUDA PagedAttention、Token Budget Sweep、FP16/BF16、CUDA Graph 与 Prefix Cache 已接通；
   各阶段结果见任务 03--08 文档。

## 复现

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda run -p /home/miniconda3/envs/zyf1 make benchmark_gpt2_serving
OMP_NUM_THREADS=16 conda run -p /home/miniconda3/envs/zyf1 \
  ./benchmark_gpt2_serving \
  --repeats 3 \
  --max-new-tokens 4 \
  --json benchmark/results/gpt2_cpu_xeon5119t_omp16.json \
  --csv benchmark/results/gpt2_cpu_xeon5119t_omp16.csv
```

原始结果：

- `benchmark/results/gpt2_cpu_xeon5119t_omp16.json`
- `benchmark/results/gpt2_cpu_xeon5119t_omp16.csv`

JSON 记录了精确 Prompt Token、生成 Token、逐请求指标、编译器、编译参数、Git commit
和每轮原始结果。三种模式生成 Token 不一致时，Benchmark 会直接失败。
