# 开发任务 05：GPU Multi-Token Prefill

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
