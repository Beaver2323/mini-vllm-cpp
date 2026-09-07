# 开发任务 07：Residual + LayerNorm 融合与 CUDA Graph

## 1. 为什么只做这两个点

任务 06 已让 GEMM 使用 FP16 Tensor Core。Nsight 中仍能看到大量 LayerNorm、Residual、
Bias、GELU 和 CUDA Launch。这个任务只回答两个问题：

1. 把相邻的 Residual Add 和 LayerNorm 合并，能否减少 Kernel？
2. 把固定 Token Batch 的整段设备计算捕获成 CUDA Graph，能否降低 CPU Launch 开销？

cuBLASLt Bias Epilogue 暂未加入，避免同时改变太多模块。

## 2. Residual + LayerNorm 融合

Transformer Block 中原来的调用是：

```text
projected = Linear(attention)
residual_b = residual_a + projected     <- Residual Kernel
normalized = LayerNorm(residual_b)      <- LayerNorm Kernel
```

融合后一个 CUDA Block 处理一行 Token：

```text
Residual 相加并写回目标精度
        ↓ __syncthreads
FP32 求均值
        ↓ __syncthreads
FP32 求方差
        ↓ __syncthreads
归一化并写 FP16 输出
```

FP16 专用路径一次加载两个元素，使用 `half2` 完成 Residual、Weight 和 Bias 读写。Residual
结果先舍入到 FP16，再转成 FP32 做 LayerNorm，保持和未融合路径相同的精度边界。

每次完整模型前向原来有 25 个 LayerNorm 和 24 个 Residual Kernel。融合后变成 1 个初始
LayerNorm和 24 个融合 Kernel，设备 Kernel 数从 49 降为 25。

### 一个必须保留的负优化结论

融合 Eager 路径并没有更快：

| 路径 | 总时间中位数 | 吞吐中位数 |
| --- | ---: | ---: |
| 未融合 Eager | 6.044 ms | 2647.036 tok/s |
| 融合 Eager | 7.599 ms | 2105.604 tok/s |

融合 Kernel 包含更多同步和寄存器状态，单个 Kernel 的执行成本超过了省下的 Launch。
因此融合默认关闭，通过 `--fusion` 显式启用。这个结果说明优化必须用端到端数据验收，
不能用“Kernel 数更少”代替性能结论。

## 3. CUDA Graph 的边界

Graph Cache 以 `total_tokens` 为 Key。Runner 的精度、模型权重、最大容量和融合开关在
构造后不变，所以同一个 Runner 内无需重复放进 Key。

动态元数据不捕获进 Graph：

```text
CPU Scheduler
  └─ H2D 更新 Token / Position / Context / Slot / Block Table
       └─ cudaGraphLaunch（读取相同设备地址中的新内容）
            └─ Embedding → 12 层 Transformer → Logits → Argmax
```

首次遇到某个 Token Batch Size 时：

1. 等待该批次元数据上传完成；
2. `cudaStreamBeginCapture`；
3. 记录所有 CUDA Kernel 和 cuBLAS 调用；
4. `cudaStreamEndCapture` 与 `cudaGraphInstantiate`；
5. 启动一次新建 Graph。

后续相同 Batch Size 只更新元数据并调用 `cudaGraphLaunch`。测试中的异长混合负载建立了
3 个 Graph，说明 Key 并不假定所有 Scheduler Step 大小相同。

Benchmark 必须复用同一个 Engine。若每次 Repeat 都重新构造 Runner，正式计时会重复
Graph Capture，测不到 Replay 收益。本任务修正为 Warmup 捕获、正式 Repeat 只 Replay。

## 4. 正确性与安全性

FP16 融合 + Graph 路径覆盖 Packed Prefill、Decode、混合 Batch、跨页和 Block 复用：

```text
max_abs_logit_error=0.117905
argmax_mismatches=0
graph_cache_size=3
Compute Sanitizer memcheck=0 errors
```

未融合回退路径继续保留。Graph 只固定执行拓扑和设备地址，Context Length、Slot Mapping
等数据每轮更新，因此不会固定请求内容。

## 5. RTX 3090 A/B Benchmark

固定 4 请求、Prompt 8/16/24/32、每请求输出 4 Token、FP16、Token Budget 64。Warmup
一次，正式重复 5 次：

| Fusion | CUDA Graph | 总时间 | 吞吐 | 相对未融合 Eager |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 关闭 | 6.044 ms | 2647.036 tok/s | 基线 |
| 开启 | 关闭 | 7.599 ms | 2105.604 tok/s | -20.5% |
| 关闭 | 开启 | 5.215 ms | 3068.092 tok/s | +15.9% |
| 开启 | 开启 | 5.016 ms | 3189.767 tok/s | **+20.5%** |

Graph 是主要收益来源。融合在 Eager 中是负优化，但在 Graph 模式下比未融合 Graph 再高
约 4.0%，所以当前推荐组合是显式启用 `--fusion --cuda-graph`。

Nsight 使用 `--cuda-graph-trace=node` 采集图内 Kernel。与任务 06 Profile 相比，模型
计算 Kernel 实例从 2,080 降至 1,840，正好减少 240 次，也就是 10 次前向中每次减少
24 次。Graph Warmup 捕获 4 个批次形状，正式阶段用 10 次 `cudaGraphLaunch` 完成执行。

## 6. 复现命令

```bash
make GPU_COMPUTE_CAPABILITY=86 \
  test_gpt2_cuda_model_runner benchmark_gpt2_cuda_serving

CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_model_runner \
  --precision fp16 --cuda-graph

CUDA_VISIBLE_DEVICES=0 ./benchmark_gpt2_cuda_serving \
  --precision fp16 --fusion --cuda-graph \
  --token-budget 64 --repeats 5 \
  --json benchmark/results/gpt2_cuda_task07_fused_graph_rtx3090.json \
  --csv benchmark/results/gpt2_cuda_task07_fused_graph_rtx3090.csv
```

原始 A/B JSON/CSV 使用 `gpt2_cuda_task07_{unfused,fused}_{eager,graph}_rtx3090`
命名。Nsight 汇总为：

- `gpt2_cuda_task07_fused_graph_nsys_cuda_gpu_kern_sum.csv`；
- `gpt2_cuda_task07_fused_graph_nsys_cuda_api_sum.csv`。

## 7. 需要掌握的结论

1. Fusion 减少 GPU Kernel 数，CUDA Graph 减少 CPU 提交调用，它们解决的不是同一层问题。
2. Graph 可以固定地址而不固定数据，动态调度元数据仍可在 Replay 前更新。
3. Graph Cache 必须按会改变 Grid、GEMM Shape 的维度分桶，本项目使用 Token Batch Size。
4. 首次 Capture 和 Instantiate 是初始化成本，不能混入稳定 Replay Benchmark。
5. 融合可能负优化，必须同时比较 Eager、Fusion、Graph 和组合路径。
