# 开发任务 06：FP16/BF16 混合精度与 Tensor Core

前置知识：[PyTorch 算子、矩阵形状与 cuBLAS 调用对照](from_pytorch/04_pytorch_to_cuda.md)。
先沿 FP32 前向走通数据流，再比较存储精度、累加精度和低精度分支。

## 1. 这次解决什么问题

任务 05 已将多个 Prefill Token 压成一个 GEMM Batch，但模型仍使用 FP32 权重、激活和
KV Cache。GPT-2 124M 的 FP32 权重约占 498 MB，线性层也没有使用 FP16 Tensor Core。

本任务为同一个 `GPT2CudaModelRunner` 增加三种运行模式：

- `FP32`：保留原始正确性基线；
- `FP16`：正式推荐路径，权重、激活、Q/K/V 和分页 KV Cache 使用半精度；
- `BF16`：实验路径，用来观察更大动态范围和更低尾数精度的影响。

最终 logits 仍保存为 FP32，便于逐词表比较、设备侧 Argmax 和误差分析。

## 2. 混合精度边界

| 数据或计算 | FP16/BF16 路径 | 原因 |
| --- | --- | --- |
| 模型权重 | FP16/BF16 存储 | 权重显存减半，GEMM 可使用 Tensor Core |
| Transformer 激活 | FP16/BF16 存储 | 降低中间结果流量和显存 |
| 分页 K/V Cache | FP16/BF16 存储 | KV Cache 是长上下文服务的重要显存项 |
| LayerNorm 求和、方差 | FP32 累加 | 避免大量低精度数相加放大误差 |
| QK 点积 | FP32 累加 | Attention Score 对误差敏感 |
| Softmax 最大值、指数和 | FP32 | 保留稳定 Softmax 的数值范围 |
| Value 加权和 | FP32 累加 | 多个历史 Token 的归约使用高精度 |
| 线性层 GEMM | 半精度输入输出、FP32 累加 | 使用 Tensor Core 并控制累计误差 |
| 最终 logits | FP32 输出 | 便于全词表对齐和稳定 Argmax |

这就是“混合精度”的核心：存储类型和累加类型可以不同。只把所有 `float` 文本替换成
`half`，会让 LayerNorm 和 Attention 归约也降精度，通常不是可靠的推理实现。

## 3. 代码设计

### 3.1 一个 Runner 保留三种模式

`GPT2CudaConfig::data_type` 在运行时选择精度。设备 Tensor Buffer 保存元素数和元素
字节数，实际前向使用 `forward<float>`、`forward<__half>` 或
`forward<__nv_bfloat16>` 模板实例。

这种设计让 Scheduler、BlockManager、ModelInput 和 Engine 完全不感知数值类型，精度
只属于执行后端。它对应 vLLM 中“控制面与设备执行面分离”的设计思想。

### 3.2 权重转换

原 checkpoint 是 FP32。Runner 初始化时先上传 FP32 权重，再用 CUDA Kernel 转成目标
类型，随后释放临时 FP32 Buffer。`weight_bytes()` 只统计长期驻留的目标精度权重。

生产系统通常会直接加载半精度 checkpoint，避免额外初始化显存和转换时间。本项目保留
llm.c 原始 checkpoint 格式，因此选择初始化时转换，逻辑更容易验证。

### 3.3 Tensor Core GEMM

FP32 路径继续调用 `cublasSgemm`。FP16/BF16 路径调用：

```cpp
cublasGemmEx(
    handle, CUBLAS_OP_T, CUBLAS_OP_N,
    output_width, batch_size, input_width,
    &alpha,
    weight, storage_type, input_width,
    input, storage_type, input_width,
    &beta,
    output, storage_type, output_width,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

代码中的激活逻辑布局是行主序 `[tokens, input_width]`，cuBLAS 默认按列主序解释内存。
通过交换矩阵顺序和转置标记，避免在每层前后显式转置。

最终词表投影同样使用半精度权重和输入，但把 C 矩阵类型设为 `CUDA_R_32F`，因此 logits
无需额外转换就能被 FP32 Argmax 和测试读取。

### 3.4 `half2` 向量化

GPT-2 Head Size 为 64，权重和激活地址满足 4 字节对齐，因此 FP16 路径可以一次处理两个
半精度元素。当前实现对以下操作使用 `half2`：

- Bias Add；
- 两次 Residual Add；
- GELU 输入输出；
- PagedAttention K/V Cache 写入；
- QK 点积加载；
- Value Cache 加载和两个通道的加权累加。

模板仍保留奇数 Head Size 的标量回退。向量化减少指令和访存事务，但 Attention 的
Softmax 与归约仍是 FP32。

## 4. 正确性结果

测试覆盖 Packed Prefill、Decode、同一步混合执行、16/17 跨页、Block 释放复用和完整
50,257 词表 logits。

| 精度 | 最大 logits 绝对误差 | Argmax 分歧 | 结论 |
| --- | ---: | ---: | --- |
| FP32 | 0.000267029 | 0 | Reference 路径通过 |
| FP16 | 0.122009 | 0 | 推荐，生成 Token 与 CPU 完全一致 |
| BF16 | 1.82688 | 1 | 实验模式，当前 GPT-2 工作负载不保证 Token 一致 |

BF16 有 8 位指数，动态范围接近 FP32，但只有 7 位显式精度；FP16 指数范围较小，却有
10 位尾数。这个 GPT-2 模型的激活没有发生 FP16 溢出，反而更需要尾数精度，所以 FP16
的 Greedy Token 更稳定。不能根据“BF16 动态范围更大”直接推导它在所有模型上更准确。

FP16 Compute Sanitizer 结果：

```text
memcheck:  ERROR SUMMARY: 0 errors
racecheck: 0 hazards displayed (0 errors, 0 warnings)
```

## 5. 显存结果

正确性测试使用 3 个 KV Block、最多 8 个 Packed Token：

| 项目 | FP32 | FP16 | 变化 |
| --- | ---: | ---: | ---: |
| 权重 | 497,903,616 B | 248,951,808 B | -50.0% |
| KV Cache | 3,538,944 B | 1,769,472 B | -50.0% |
| 激活总量 | 1,978,368 B | 1,794,048 B | -9.3% |

激活没有整体减半，是因为最终 `[max_tokens, padded_vocab_size]` logits 仍为 FP32，并且
在 GPT-2 中词表矩阵远大于隐藏维度。如果以后只为每个请求的采样位置计算 logits，或把
非采样行裁掉，激活显存还能继续下降。

## 6. RTX 3090 Benchmark

固定工作负载为 4 个请求，Prompt 长度 8/16/24/32，每个请求输出 4 Token；每档先预热
一次，再正式运行 3 次。FP16 的所有 16 个生成 Token 均与独立 CPU 完整前缀 Reference
一致。

| Token Budget | 总时间中位数 | 吞吐中位数 | TTFT P50/P95 | TPOT P50/P95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 104.826 ms | 152.634 tok/s | 46.353 / 95.461 ms | 1.134 / 1.147 ms |
| 4 | 29.263 ms | 546.757 tok/s | 12.137 / 24.842 ms | 1.154 / 1.170 ms |
| 8 | 20.820 ms | 768.499 tok/s | 8.198 / 16.343 ms | 1.465 / 1.483 ms |
| 16 | 14.648 ms | 1092.322 tok/s | 5.247 / 9.841 ms | 1.695 / 1.735 ms |
| 32 | 7.366 ms | 2172.184 tok/s | 1.933 / 3.708 ms | 1.194 / 1.346 ms |
| 64 | 5.842 ms | 2738.953 tok/s | 1.289 / 2.372 ms | 1.127 / 1.274 ms |

在当前同一提交 `bcf84ea` 上重跑 FP32 Budget 64，结果为 `1866.201 tok/s`、`8.574 ms`。
FP16 相对这个同版本基线吞吐提升 **46.8%**，总耗时下降 **31.9%**。相对任务 05 保存的
FP32 基线 `1838.148 tok/s`，提升为 49.0%。

小 Token Budget 的收益不稳定，因为每次 GEMM 很小，Launch、LayerNorm、PagedAttention
和 Argmax 占比更高。Budget 32/64 才能更充分使用 Tensor Core。这也是服务系统需要
Continuous Batching 和 Token Budget 的原因：低精度硬件能力要有足够矩阵规模才能转化
为端到端收益。

## 7. Nsight Systems 证据

Budget 64 Profile 包含一次 Warmup 和一次正式运行。Kernel 名中直接出现：

- `ampere_fp16_s16816gemm...`；
- `cutlass_80_tensorop_f16_s16816gemm...`；
- `sm80_xmma_gemm_f16f16...tensor16x8x16...`。

这些名字说明 cuBLAS 选择了 Ampere FP16 Tensor Core Kernel。Profile 共记录 2,082 次
Kernel Launch、12.406 ms GPU Kernel 时间。其中两次初始化权重转换占 1.750 ms；正式
Benchmark 计时排除了 Runner 初始化，不能把这部分算进端到端延迟。

在低精度后，仍可看到 250 次 LayerNorm、480 次 Bias、240 次 Residual、120 次 GELU
以及 120 次 PagedAttention。下一阶段应优先融合 Bias/Residual/LayerNorm/GELU，并为
固定 Token Bucket 捕获 CUDA Graph，进一步减少小 Kernel 和 CPU Launch 开销。

## 8. 复现命令

```bash
make GPU_COMPUTE_CAPABILITY=86 \
  test_gpt2_cuda_model_runner benchmark_gpt2_cuda_serving

CUDA_VISIBLE_DEVICES=0 \
  ./test_gpt2_cuda_model_runner --precision fp16

CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck \
  ./test_gpt2_cuda_model_runner --precision fp16
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck \
  ./test_gpt2_cuda_model_runner --precision fp16

CUDA_VISIBLE_DEVICES=0 ./benchmark_gpt2_cuda_serving \
  --precision fp16 --token-budget 64 --repeats 3 \
  --json benchmark/results/gpt2_cuda_fp16_budget64_rtx3090.json \
  --csv benchmark/results/gpt2_cuda_fp16_budget64_rtx3090.csv
```

原始数据：

- `benchmark/results/gpt2_cuda_fp16_budget{1,4,8,16,32,64}_rtx3090.json`；
- 对应的逐次运行 CSV；
- `benchmark/results/gpt2_cuda_fp16_budget_sweep_rtx3090.csv`；
- `benchmark/results/gpt2_cuda_fp32_task06_budget64_rtx3090.json`；
- `benchmark/results/gpt2_cuda_fp16_nsys_cuda_gpu_kern_sum.csv`；
- `benchmark/results/gpt2_cuda_fp16_nsys_cuda_api_sum.csv`。

## 9. 面试需要讲清楚的四个问题

1. **FP16 存储为什么还叫 FP32 累加？** 输入和输出 Tensor 可以是半精度，但 Tensor
   Core 内部乘加累加器及 LayerNorm/Attention 归约使用 FP32。
2. **为什么 FP16 在这里比 BF16 更稳定？** 当前模型没有溢出，误差更受尾数位数影响；
   FP16 有更多尾数位。
3. **为什么显存没有全部减半？** 最终全 Token、全词表 logits 为 FP32，是激活内存的
   主要部分。
4. **为什么 Tensor Core 生效却只有 46.8% 端到端提升？** 推理还包含 Attention、归约、
   Argmax、许多小 Kernel 和 Launch；小 Batch GEMM 也无法达到峰值吞吐。
