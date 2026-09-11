# 任务 10：测清 Prefix Cache 的实际收益

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
