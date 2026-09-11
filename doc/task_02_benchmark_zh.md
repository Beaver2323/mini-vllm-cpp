# 开发任务 02：建立可复现的推理 Benchmark

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
