# 任务 10：测清 Prefix Cache 的实际收益

任务 08 证明缓存正确，本任务证明它在什么条件下节省了多少计算。这里会逐段读一份不到百余行的 Benchmark，让你知道实验变量与计时边界究竟在哪里。

学习目标：能够自己解释 off、miss、hit 三种模式，并从 Prompt 长度、命中页数推导 scheduled_tokens 的精确期望值。

## 1. 为什么不能只测“开缓存前后”

| 模式 | 缓存开关 | 目标请求到达前的缓存状态 | 主要回答的问题 |
| --- | --- | --- | --- |
| off | 关闭 | 无缓存 | 基线开销多少 |
| miss | 开启 | 先清空 | 查询与登记元数据有多少开销 |
| hit | 开启 | 先用 seed 建立公共前缀 | 命中后少计算多少、延迟改善多少 |

只比较 off 与 hit 会把两件事混在一起：开关本身的开销，以及确实存在公共前缀带来的收益。miss 帮助解释没有复用机会时的行为。

这份实验是可控微基准，所有 hit 都人为准备好；它没有测真实线上请求分布中的自然命中率。

## 2. 先按调用顺序打开源码

完整程序是 [benchmark_gpt2_cuda_prefix_cache.cu](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu)。按下面顺序读，比从 CSV 列名开始猜更快：

```text
main
  ├─ 构造 off Engine 与 cached Engine
  ├─ 对 prefix=16/64/128/256
  │   ├─ 构造 seed、目标 Prompt
  │   ├─ 用 CPU 完整前缀计算目标输出参考
  │   └─ warmup + 正式 repeats
  │       ├─ clear cache
  │       ├─ hit 模式：先运行 seed（目标计时之外）
  │       ├─ measure 目标请求
  │       ├─ 核对 Token、hit_blocks、scheduled_tokens
  │       └─ 写入一条 CSV 原始记录
  └─ 完成
```

这里 Engine 在外层创建并复用，初始化权重与设备内存不在目标请求的计时区间内。

## 3. 运行配置必须先看清

源码：[benchmark/benchmark_gpt2_cuda_prefix_cache.cu，第 45—49 行](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu#L45)。以下为当前文件的原样摘录。

```cpp
        GPT2CudaConfig config{model.config.max_seq_len, model.config.vocab_size,
            model.config.padded_vocab_size, model.config.num_layers, model.config.num_heads,
            model.config.channels, CudaDataType::FP16, false, false, true};
        GPT2CudaEngine off(config, model.params_memory, model.num_parameters, 40, {1, 272}, 272, false);
        GPT2CudaEngine cached(config, model.params_memory, model.num_parameters, 40, {1, 272}, 272, true);
```

本实验用 FP16、关闭残差融合、关闭 CUDA Graph、开启采样行裁剪。两个 Engine 的主要配置一致，只在 Prefix Cache 开关上不同。

最大并发请求数是 1，token budget 与最大上下文为 272。目标最长 Prompt 为 258，生成 4 个实际最多处理 261，容量足够；没有把页不足或排队竞争引入这个实验。

不要把这个结果说成“高并发服务器的 Prefix Cache 收益”。它针对单个目标请求的前缀复用，目的是先把因果关系测清楚。

## 4. seed 为什么是 prefix + 1

源码：[benchmark/benchmark_gpt2_cuda_prefix_cache.cu，第 59—64 行](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu#L59)。以下为当前文件的原样摘录。

```cpp
        for (int prefix : {16, 64, 128, 256}) {
            std::vector<int> seed(prefix + 1);
            seed[0] = 50256;
            for (int i = 1; i <= prefix; ++i) seed[i] = 100 + (i * 7919) % 50000;
            std::vector<int> prompt(seed.begin(), seed.begin() + prefix);
            prompt.push_back(1234); prompt.push_back(4321);
```

记公共前缀长度为 S，且 S 是 16 的倍数。seed 长 S+1，目标 Prompt 长 S+2，两者前 S 个 Token 相同，目标后面接自己的两个后缀 ID。

任务 08 的可缓存完整页数为 `floor((prompt_len−1)/16)`。因此：

```text
seed_len = S+1
可登记块数 = floor(S/16) = S/16
目标输入 = 公共前缀 S + 私有后缀 2
命中后只剩 2 个 Prompt Token 要计算
```

如果 seed 只有 S 个 Token，策略会保留最后一整块重算，最多登记 `S/16−1` 块，实验就测不到预期的完整 S 长复用。

seed 的生成数设为 1，使它只完成 Prompt、产生一个输出并结束。它的生成 ID 不属于公共前缀，也不会成为目标请求的输入。

## 5. CPU reference 与测量严格分开

源码：[benchmark/benchmark_gpt2_cuda_prefix_cache.cu，第 65—73 行](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu#L65)。以下为当前文件的原样摘录。

```cpp
            // 完整 CPU Greedy 独立参考，计时之外。
            auto tokens = prompt;
            std::vector<int> expected;
            for (int i = 0; i < 4; ++i) {
                gpt2_forward_dense_with_workspace(&model, tokens.data(), 1, tokens.size(), &reference);
                const auto* logits = reference.acts().logits + (tokens.size() - 1) * model.config.padded_vocab_size;
                const int token = std::max_element(logits, logits + model.config.vocab_size) - logits;
                expected.push_back(token); tokens.push_back(token);
            }
```

参考模型每次按当前完整前缀执行，选择最后一行真实词表范围内的 argmax，再追加 Token。这样得到 4 个固定期望输出。

这些计算发生在目标计时之前。它比仅比较 off/hit 两个 GPU 模式更独立：如果两者共享了同一个 GPU 逻辑错误，只互相比较可能一起出错而看不出来。

参考只验证该实验中的 greedy 结果，不代表完成了所有模型、长上下文与随机采样分布的精度验证。

## 6. measure：观察量怎样采集

源码：[benchmark/benchmark_gpt2_cuda_prefix_cache.cu，第 20—36 行](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu#L20)。以下为当前文件的原样摘录。

```cpp
Observation measure(GPT2CudaEngine& engine, std::uint64_t id, const std::vector<int>& prompt,
                    std::size_t new_tokens = 4) {
    const auto hit_before = engine.prefix_cache_hit_blocks();
    const auto start = Clock::now();
    auto request = engine.add_request(id, prompt, {new_tokens, -1, true});
    Observation result;
    while (!engine.is_finished()) {
        auto step = engine.step();
        result.scheduled_tokens += step.num_batched_tokens;
        if (request->num_completion_tokens() == 1)
            result.ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    result.hit_blocks = engine.prefix_cache_hit_blocks() - hit_before;
    result.generated.assign(request->token_ids().begin() + prompt.size(), request->token_ids().end());
    return result;
}
```

逐段阅读：

1. `hit_before` 在请求提交前保存，因为命中统计是 Engine 累计值。
2. `start` 在 `add_request` 之前，所以目标请求构造和入队算入时间。
3. 每次 `step` 累加实际输入 Token 数，而不是把请求当前总长度重复相加。
4. 输出数量第一次达到 1 时记录 TTFT。
5. Engine 全部结束后记录总时间，提取 Prompt 之后的生成 ID。
6. 命中数取后减前，避免 seed 或上一次实验污染当前计数。

在这个单请求、正常连续推进的设置里，首 Token 出现之后的每轮 Decode 都会增加输出数，所以 `==1` 能记录首 Token 时刻。若将来改成复杂多请求测量，应使用一次性标志，避免请求等待时重复覆盖 TTFT；不能直接把这段观察代码照搬到所有调度场景。

## 7. 手算 scheduled_tokens，先验证省掉的工作量

目标 Prompt 长 S+2，输出 4 个，使用 KV 时总需计算 `S+2+3=S+5` 个输入 Token。

hit 模式跳过 S 个已缓存 Token，只计算 2 个后缀及 3 个 Decode 输入，所以始终为 5。

| 公共前缀 S | Prompt 长度 | 命中块数 | off/miss 输入数 | hit 输入数 | 跳过输入数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 18 | 1 | 21 | 5 | 16 |
| 64 | 66 | 4 | 69 | 5 | 64 |
| 128 | 130 | 8 | 133 | 5 | 128 |
| 256 | 258 | 16 | 261 | 5 | 256 |

这个确定性的计数比一次时间波动更适合先验证功能。如果 hit 显示更快，但 scheduled_tokens 仍等于 S+5，应怀疑缓存没有真正跳过计算，时间差可能只是噪声。

相反，即使输入数从 261 降到 5，延迟也不应预期降低 52 倍：剩余 Query 的 Attention 仍要读取公共前缀 KV，且其他固定开销仍存在。

## 8. 模式切换与 warmup 逐段解释

源码：[benchmark/benchmark_gpt2_cuda_prefix_cache.cu，第 74—92 行](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu#L74)。以下为当前文件的原样摘录。

```cpp
            // 每种模式一次预热，再轮换模式测量；所有 seed 和清理均在计时区间之外。
            for (int repeat = -1; repeat < repeats; ++repeat) {
                for (int mode = 0; mode < 3; ++mode) {
                    auto& engine = mode == 0 ? off : cached;
                    engine.clear_prefix_cache();
                    if (mode == 2) measure(engine, id++, seed, 1);
                    auto result = measure(engine, id++, prompt);
                    if (result.generated != expected) throw std::runtime_error("prefix benchmark CPU token mismatch");
                    const auto expected_hits = mode == 2 ? prefix / 16 : 0;
                    const auto expected_scheduled = prompt.size() + 3 - expected_hits * 16;
                    if (result.hit_blocks != static_cast<std::size_t>(expected_hits) ||
                        result.scheduled_tokens != expected_scheduled)
                        throw std::runtime_error("prefix benchmark unexpected cache reuse");
                    if (repeat < 0) continue;
                    const char* name = mode == 0 ? "off" : (mode == 1 ? "miss" : "hit");
                    csv << props.name << ",fp16,false,false,true," << prefix << ',' << prompt.size()
                        << ',' << name << ',' << repeat << ',' << result.scheduled_tokens << ','
                        << result.hit_blocks << ',' << result.ttft_ms << ',' << result.total_ms << ','
                        << (result.total_ms - result.ttft_ms) / 3 << ',' << 4000 / result.total_ms << '\n';
```

每次 mode 开始先 clear，确保 miss 确实无缓存。hit 模式随后单独跑 seed，再开始目标 `measure`。seed 和 clear 都在目标计时区间之外。

`repeat=-1` 是 warmup：照样执行、照样验证，但不写 CSV。正式 repeat 从 0 开始。

模式在每个 repeat 内按 off、miss、hit 的固定顺序执行，这是一种交错测量，但不是随机顺序或每轮反转顺序。温度、频率等系统漂移仍可能影响小幅差异，因此应保留原始点并避免对很小的变化过度解释。

为了量化真实整体成本，可以另设计“seed+target 总工作量”实验；当前 CSV 的 hit 数字明确表达**缓存已经准备好之后的目标请求时间**，不是缓存建设免费。

## 9. TTFT、TPOT 与吞吐的具体分母

目标固定生成 4 个 Token，因此代码写：

```text
TPOT = (total_ms - ttft_ms) / 3
output_tokens_per_second = 4000 / total_ms
```

`4000` 来自 `4 tokens * 1000 ms/s`。如果以后把 new_tokens 改成 8，却不改这两处，程序仍可能运行，但指标就错了。

教学时间线：目标 TTFT=2 ms，完成=5 ms，则 TPOT=1 ms，生成吞吐=800 tok/s。不是 `1/2ms=500 tok/s`，也不是 `4/3ms`。

Prefix Cache 直接省掉的是公共前缀的前向计算，通常最容易在 TTFT 观察到；后续 Decode 仍需关注完整历史，TPOT 不保证按命中长度同比降低。

## 10. 从 CSV 自己算一次中位数

可以用标准库读取仓库已保存的原始点，不需要再启动 GPU：

```python
# 在仓库根目录运行的教学分析代码。
import csv
import statistics
from collections import defaultdict

groups = defaultdict(list)
with open('benchmark/results/task09_11/prefix.csv', newline='') as f:
    for row in csv.DictReader(f):
        key = (int(row['prefix_tokens']), row['mode'])
        groups[key].append(float(row['ttft_ms']))
for key in sorted(groups):
    print(key, 'n=', len(groups[key]), 'median_ms=', statistics.median(groups[key]))
```

先核对每组样本数，再计算中位数。百分比下降为 `(miss−hit)/miss*100%`；加速倍数为 `miss/hit`。两者单位和措辞不同。

仓库 256 前缀的历史 TTFT 约为 miss 4.057 ms、hit 1.541 ms，下降约 62%。这只代表记录中的设备、模型与固定工作负载，不能直接推广为全部请求吞吐增加 62%。

## 11. 诊断表：结果不符合预期时看哪里

| 观察 | 首先检查 |
| --- | --- |
| hit_blocks=0 | seed 是否运行、前 S 个 Token 是否真的相同 |
| 少命中一页 | seed 是否误写成 S 而不是 S+1 |
| 命中数正常但输出错误 | 缓存 key 是否完整、共享尾页是否被覆盖 |
| scheduled_tokens 正常但时间接近 | 剩余 Attention/固定开销占比、样本波动 |
| 第二轮 miss 竟然命中 | clear 是否在每次 mode 前执行 |
| hit_blocks 大于期望 | 是否忘记用累计计数差值 |

不要看到性能不明显就立即增加功能。先把可复现的计数、输出、时间线串起来，这是你进入推理优化岗位最值得掌握的实验方法。

## 12. 练习与答案

**题 1：S=64，目标生成 6 个，off 与 hit 的输入数是多少？**

答案：Prompt=66，off=`66+5=71`，hit=`2+5=7`。命中块数仍是 4。

**题 2：clear 缓存后累计 hit 计数不归零，如何统计目标请求？**

答案：保存提交前计数，用完成后的计数减去它。

**题 3：测量 hit 时把 seed 时间算进去，会回答什么不同问题？**

答案：会更接近“为这组请求建立并使用缓存的总成本”，而不是已热缓存下单个目标请求的收益。两个指标都可以研究，但应分别命名。

**题 4：miss 与 off 有很小差异，能断定 map 查询是瓶颈吗？**

答案：不能，仅有总时间不足以定位。需要重复点、配置一致性和更细的 CPU 开销证据。

**题 5：hit 只计算两个 Prompt 后缀，Attention 可否只看这两个 Token？**

答案：不能。它们仍然需要读取公共前缀的 KV；省掉的是公共前缀的重复前向，而不是删除上下文。

下一篇：[任务 11：双 GPU 的功能性 PD 分离](task_11_pd_disaggregation_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[指标与实验手册](from_pytorch/06_labs_and_answers.md)。
先自己画一条 Token 输出时间线，再理解本实验的 TTFT 与 TPOT。

任务 08 已实现缓存功能。本任务补齐可复现的性能对照，让“跳过前缀”对应到实际调度量、
首 Token 延迟和完整生成耗时。先完成任务 08 的引用计数学习，再运行本篇实验。

## 1. 每一个学习点到哪里看

| 要学什么 | 代码位置 | 调用点或消费者 |
| --- | --- | --- |
| 构造关闭和开启缓存的两套引擎 | [benchmark_gpt2_cuda_prefix_cache.cu](../benchmark/benchmark_gpt2_cuda_prefix_cache.cu)，`main` 中 `off`、`cached` | 下面的 mode 循环 |
| 构造相同前缀、不同后缀 | 同文件 `for (int prefix : ...)` | `seed` 先执行、`prompt` 再测量 |
| 清理缓存但保留模型和 Buffer | [GPT2CudaEngine::clear_prefix_cache](../mini_vllm/cuda/gpt2_cuda_engine.hpp) | Benchmark 每次测量前调用 |
| 实际释放缓存引用 | [BlockManager::clear_prefix_cache](../mini_vllm/block_manager.hpp) | 上面的 Engine 包装接口 |
| 计时范围和采样 | Benchmark `measure` | `main` 中三种模式都走同一个函数 |
| 缓存实际跳过多少 Token | `Observation::scheduled_tokens`、`hit_blocks` | `expected_scheduled` 断言 |
| CPU 正确性参考 | `gpt2_forward_dense_with_workspace` 的调用 | Benchmark 在 warmup/正式计时外计算 |
| 查看原始逐轮记录 | [prefix.csv](../benchmark/results/task09_11/prefix.csv) | 按 prefix 和 mode 分组计算中位数 |

定位命令：

```bash
rg -n 'measure\(|clear_prefix_cache|expected_scheduled|hit_before|repeat = -1' \
  benchmark/benchmark_gpt2_cuda_prefix_cache.cu
```

## 2. 为什么必须有三组

| 模式 | 缓存开关 | 测量前操作 | 测量请求执行什么 |
| --- | --- | --- | --- |
| off | 关闭 | 无种子请求 | 完整 Prompt |
| miss | 开启 | 清空缓存 | 完整 Prompt，同时注册缓存 |
| hit | 开启 | 清空后先运行共享前缀 seed | 只计算未命中的后缀 |

off 对比 miss 能看到缓存查找、Key 构造和引用维护的开销；miss 对比 hit 才能观察复用的收益。
只比较第一次请求与第二次请求会把 GPU 预热、cuBLAS 初始化等成本混进结果。

三组统一使用 GPT-2 124M、FP16、采样行裁剪开启、Fusion 关闭、CUDA Graph 关闭、Batch=1、
Token Budget=272、Context 容量 272、KV Pool=40 页。Graph 关闭是为了保持本任务只观察缓存效果。

## 3. 从输入构造到真实 KV 命中

Benchmark 针对前缀长度 16、64、128、256 分别构造：

```cpp
std::vector<int> prompt(seed.begin(), seed.begin() + prefix);
prompt.push_back(1234);
prompt.push_back(4321);
```

seed 长度为 `prefix + 1`，测量 Prompt 长度为 `prefix + 2`。它们前 prefix 个 Token 相同，后缀
不同。种子请求只生成 1 个 Token，主要目的是把完整前缀页注册进缓存。

实际调用点：

```cpp
engine.clear_prefix_cache();
if (mode == 2) measure(engine, id++, seed, 1);
auto result = measure(engine, id++, prompt);
```

`clear_prefix_cache` 和 seed 执行都发生在测量请求的计时区间之外。调用 Engine 清理接口会进一步
调用 BlockManager，减少缓存自己的引用计数，引用归零的页放回 free list。模型权重、CUDA
上下文、cuBLAS Handle 和激活 Buffer 不重建。

缓存计数器是累计值，清理缓存不会清零累计命中数。因此每次测量通过“前后差值”取本请求的
命中页数，不能把累计数当成单次数据。

## 4. 调用链如何证明跳过了计算

```text
measure
  └─ engine.add_request
  └─ engine.step
       └─ Scheduler::schedule
            └─ try_schedule
                 └─ apply_prefix_cache
                      ├─ 增加命中页引用
                      ├─ 填入新请求的 Block Table
                      └─ mark_computed(hit_blocks × 16)
                 └─ pending_tokens 只剩 2
       └─ Runner：只打包这 2 个 Token
       └─ PagedAttention：通过共享页表读取完整历史 KV
       └─ commit：得到第一个输出 Token
```

假设 prefix=128：off/miss 第一步执行 130 个输入 Token，hit 只执行 2 个。后面都执行 3 轮
Decode，生成总共 4 个新 Token。完整计时窗口中的调度量应是：

```text
off / miss：130 + 3 = 133
hit：         2 + 3 =   5
hit_blocks：128 / 16 = 8
```

代码检查：

```cpp
const auto expected_hits = mode == 2 ? prefix / 16 : 0;
const auto expected_scheduled = prompt.size() + 3 - expected_hits * 16;
if (result.hit_blocks != static_cast<std::size_t>(expected_hits) ||
    result.scheduled_tokens != expected_scheduled)
    throw std::runtime_error("prefix benchmark unexpected cache reuse");
```

只看延迟下降不够，这两个计数能确认节省来自实际跳过前缀。

## 5. 如何定义计时边界

`measure` 的关键顺序：

```cpp
const auto hit_before = engine.prefix_cache_hit_blocks();
const auto start = Clock::now();
auto request = engine.add_request(id, prompt, {new_tokens, -1, true});
while (!engine.is_finished()) {
    auto step = engine.step();
    result.scheduled_tokens += step.num_batched_tokens;
    if (request->num_completion_tokens() == 1)
        result.ttft_ms = /* Clock::now() - start */;
}
```

Runner 返回前会同步自己的 CUDA Stream，所以这里的 CPU 时钟覆盖实际完成的 GPU 工作，
不是只测 Kernel Launch。CSV 中的定义如下：

| 字段 | 本实验含义 |
| --- | --- |
| `ttft_ms` | 从 add_request 前到首 Token 生成完成，含本地请求构造与调度 |
| `total_ms` | 同一起点到 4 个 Token 全部生成完成 |
| `tpot_ms` | `(total_ms - ttft_ms) / 3`，首 Token 后每 Token 平均耗时 |
| `output_tokens_per_second` | `4 × 1000 / total_ms`，单请求窗口吞吐 |
| `scheduled_tokens` | 所有 step 实际提交的输入 Token 数 |
| `hit_blocks` | 本请求共享的完整前缀页数 |

这里没有 HTTP、分词、网络传输或真实线上到达时间。不要把这个 Batch=1 的窗口吞吐直接
等同于在线服务容量。

## 6. 预热、重复和正确性

循环从 `repeat = -1` 开始：三种模式各预热一次，不写入正式 CSV；随后各测 7 次。
每次都清理状态并按模式重新播种，因此 miss 不会因为上一轮刚测过就变成 hit。

每个测量 Prompt 的全部 4 个 Greedy Token 都与 CPU 完整前缀参考对齐，包括后续 Decode。
CPU 参考在正式计时外计算，不会污染样本窗口。复现失败时优先检查：

1. off/miss/hit 是否生成相同结果。
2. hit 是否真的命中 `prefix / 16` 页。
3. 所有模式是否使用同样的精度与采样行开关。
4. GPU 是否有其他工作负载，测量时是否发生温度/频率变化。

这些检查对应本实验实际依赖的条件。短请求中，CUDA Launch 和 CPU 开销占比可能较高，即使
跳过很多 Token，也不一定按跳过比例降低 TTFT。

## 7. 运行与读结果

```bash
conda activate zyf1
make benchmark_gpt2_cuda_prefix_cache GPU_COMPUTE_CAPABILITY=86
OMP_NUM_THREADS=8 ./benchmark_gpt2_cuda_prefix_cache \
  benchmark/results/prefix_cache_comparison.csv 7
```

两个位置参数分别为输出文件和重复次数。目录须已存在。项目正式数据与汇总见
[任务 09—11 结果](../benchmark/results/task09_11/README.md)。正式 CSV 保存每一轮数据，
不是只保留最好的值。

建议先选 prefix=128，手算三组 `scheduled_tokens`，再比较 TTFT 中位数，最后看 TPOT 是否
基本稳定。你应能解释：缓存主要减少 Prefill 工作量，Decode 仍要访问同样长度的历史 KV。

## 8. 练习与面试表述

练习一：把 seed 头部一个 Token 改掉，预测缓存命中数量，再运行验证。由于 Key 包含完整
历史前缀，后续相同片段不能直接复用。

练习二：尝试 15/16/17 Token 的共享前缀，解释完整页限制导致的台阶效应。

可以据实表述：“实现完整页 Prefix Cache，并设计关闭、未命中、命中三组对照，通过调度
Token 数、命中页数和 CPU 输出一致性验证实际 KV 复用，测量不同前缀长度下的 TTFT。”
具体速度提升只能引用本机对应工作负载的数据。
