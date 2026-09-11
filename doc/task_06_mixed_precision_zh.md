# 开发任务 06：FP16/BF16 混合精度与 Tensor Core

从 PyTorch 迁移到手写 Runner，最容易误解的是“改成 half”只需换一个类型。本任务把参数上传、GEMM、归约、输出与测试的精度边界分别讲清楚。

学习目标：能沿着一条前向链标注每块内存的 dtype、每次累加的 dtype，以及什么时候发生舍入。

## 1. 先阅读配置与实际分派

入口在 [GPT2CudaConfig](../mini_vllm/cuda/gpt2_cuda_model_runner.cuh)，实现从 [Runner::run](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L776) 进入对应模板：

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 776—781 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L776)。以下为当前文件的原样摘录。

```cpp
        const std::vector<int> token_samples =
            config_.data_type == CudaDataType::FP16
                ? forward<__half>(input)
                : (config_.data_type == CudaDataType::BF16
                    ? forward<__nv_bfloat16>(input)
                    : forward<float>(input));
```

这里不是 PyTorch autocast。autocast 根据算子策略选择精度；当前 C++ 实现由配置选择整条模板前向，再由各个算子显式规定存储、输出与累加类型。

| 对象 / 运算 | FP32 路径 | FP16 路径 | BF16 路径 |
| --- | --- | --- | --- |
| 权重、KV、大部分 hidden | float | half | bfloat16 |
| LayerNorm sum/variance | float | float | float |
| Attention 点积、softmax、V 累加 | float | float | float |
| 一般 GEMM 输出 | float | half | bfloat16 |
| LM head logits | float | float | float |
| 元数据、采样 ID | int | int | int |

所以“FP16 推理”是对主要存储与矩阵输入路径的概括，不能说所有缓冲都只占两字节。

## 2. 类型擦除缓冲没有完整 dtype 检查

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 96—120 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L96)。以下为当前文件的原样摘录。

```cpp
class DeviceTensorBuffer {
public:
    DeviceTensorBuffer(std::size_t count, std::size_t element_size)
        : count_(count), element_size_(element_size) {
        if (count == 0 || (element_size != sizeof(float) &&
                          element_size != sizeof(__half))) {
            throw std::invalid_argument("invalid CUDA tensor buffer");
        }
        check_cuda(cudaMalloc(&pointer_, bytes()), "cudaMalloc tensor");
    }

    ~DeviceTensorBuffer() {
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

    DeviceTensorBuffer(const DeviceTensorBuffer&) = delete;
    DeviceTensorBuffer& operator=(const DeviceTensorBuffer&) = delete;

    template <typename T>
    T* get() {
        if (sizeof(T) != element_size_) {
            throw std::logic_error("CUDA tensor type does not match storage");
        }
        return static_cast<T*>(pointer_);
    }
```

`element_size` 决定字节数，`get<T>()` 检查 `sizeof(T)` 是否匹配。这能发现把 2 字节内存当 4 字节元素访问，却不能区分 FP16 与 BF16，因为两者大小相同。

FP16/BF16 的语义正确性还依赖前向模板分派、初始化转换和 cuBLAS `storage_type` 一致。只检查字节大小不足以证明 dtype 正确。

PyTorch Tensor 同时保存 dtype、device、layout 等元数据；这里采用更小的封装，要求调用点承担更多约束。读这段代码时要学会识别“封装检查了什么”和“调用者必须保证什么”。

## 3. 权重初始化：转换发生一次，不在每轮执行

从构造函数跳到 [initialize_parameters](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L941)，先读数据流：

```text
host FP32 checkpoint
  ├─ FP32 配置：直接 H2D 到参数池
  └─ FP16/BF16 配置：H2D 到临时 FP32 缓冲
                     → 转换 kernel → 目标参数池
                     → 同步后释放临时缓冲
```

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 952—977 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L952)。以下为当前文件的原样摘录。

```cpp
        DeviceBuffer<float> fp32_parameters(num_parameters_);
        check_cuda(
            cudaMemcpyAsync(
                fp32_parameters.get(), host_parameters,
                fp32_parameters.bytes(), cudaMemcpyHostToDevice,
                stream_.get()),
            "copy GPT-2 weights before FP16 conversion");
        if (config_.data_type == CudaDataType::FP16) {
            float_to_half_kernel<<<
                blocks_for(static_cast<int>(num_parameters_)), kThreads, 0,
                stream_.get()>>>(
                parameters_.get<__half>(), fp32_parameters.get(),
                num_parameters_);
            check_last_kernel("convert GPT-2 weights to FP16");
        } else {
            float_to_bfloat16_kernel<<<
                blocks_for(static_cast<int>(num_parameters_)), kThreads, 0,
                stream_.get()>>>(
                parameters_.get<__nv_bfloat16>(), fp32_parameters.get(),
                num_parameters_);
            check_last_kernel("convert GPT-2 weights to BF16");
        }
        check_cuda(
            cudaStreamSynchronize(stream_.get()),
            "finish reduced-precision weight conversion");
    }
```

`fp32_parameters` 是局部临时对象，转换写入的 `parameters_` 是 Runner 长期持有的目标池。FP16 与 BF16 分支明确选择不同转换 kernel，最后同步保证临时输入不被提前释放。

参数视图 `point_parameters<T>` 在一个连续参数池中按张量元素数切分，每个切分都使用 T 的指针算术。因此转换后参数张量的元素数量与顺序不变，改变的是每元素字节数及可表示数值。

为什么临时 FP32 buffer 不能提前释放？转换 kernel 还会读取它。C++ 局部变量退出作用域会触发析构，但 CUDA launch 是异步的；必须在读取结束后才能归还内存。

初始化时间和临时显存峰值与稳态每步时间是不同指标。文末 Benchmark 排除模型初始化，不能拿它说明模型加载也按同样比例加速。

## 4. 普通 GEMM：输入 half，累加 float，输出 half

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 998—1008 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L998)。以下为当前文件的原样摘录。

```cpp
        } else {
            constexpr cudaDataType_t storage_type =
                std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
            check_cublas(cublasGemmEx(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, storage_type, input_width,
                input, storage_type, input_width, &beta,
                output, storage_type, output_width,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx FP16 Tensor Core");
```

这几行同时表达三种类型：

1. weight 与 input 使用 `storage_type`，由 T 决定 FP16 或 BF16。
2. `CUBLAS_COMPUTE_32F` 指定计算类型。
3. output 仍是 `storage_type`，累加结果在写回时转成低精度。

因此即使 GEMM 内部采用 FP32 累加，中间 hidden 的舍入仍会随层数传播。不能认为“累加 float”就与全 FP32 模型完全一致。

`CUBLAS_GEMM_DEFAULT_TENSOR_OP` 是算法选择相关参数，实际选用的 kernel 还受形状和设备影响；应结合 profiler 观察，不能只凭枚举名保证每个矩阵尺寸都达到 Tensor Core 峰值。

源码中的诊断字符串含 `FP16`，该分支也处理 BF16。排错时以 `storage_type` 和实际配置为准，不要被错误信息的历史措辞误导。

## 5. LM head 为什么保留 FP32 logits

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1038—1048 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1038)。以下为当前文件的原样摘录。

```cpp
        } else {
            constexpr cudaDataType_t storage_type =
                std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
            check_cublas(cublasGemmEx(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, storage_type, input_width,
                input, storage_type, input_width, &beta,
                logits_.get(), CUDA_R_32F, output_width,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx FP16 logits");
```

普通 GEMM 的 output 类型与输入相同；这一处明确把 logits 输出设为 `CUDA_R_32F`。最终 argmax 在这些 float 分数上比较。

这能避免 logits 写回时再增加一次低精度量化，但前面权重和 hidden 已有的误差不会被恢复。将半精度数转回 float 只提高后续计算的表示能力，不能找回已经舍入掉的信息。

任务 09 进一步只投影需要采样的 R 行，显著减少 FP32 logits 缓冲。文末任务 06 的历史激活容量是在更早的全行投影阶段测得，不能直接作为当前默认内存占用。

## 6. LayerNorm：存储低精度，统计量用 float

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 347—360 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L347)。以下为当前文件的原样摘录。

```cpp
        static_cast<std::size_t>(row) * channels;
    float local_sum = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        local_sum += to_float(row_input[channel]);
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float mean = reduction[0] / channels;
    __syncthreads();
```

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 362—385 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L362)。以下为当前文件的原样摘录。

```cpp
    float local_variance = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        const float shifted = to_float(row_input[channel]) - mean;
        local_variance += shifted * shifted;
    }
    reduction[thread] = local_variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float inverse_stddev =
        rsqrtf(reduction[0] / channels + 1e-5f);
    __syncthreads();

    T* row_output = output +
        static_cast<std::size_t>(row) * channels;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        row_output[channel] = from_float<T>(
            (to_float(row_input[channel]) - mean) * inverse_stddev *
                to_float(weight[channel]) +
            to_float(bias[channel]));
```

统计量分两遍计算：先均值，再求 `(x−mean)²` 的平均值。最后加 epsilon、求平方根倒数、应用 weight 与 bias，再转回 T。

PyTorch 教学对照：

```python
# 展示精度边界，不承诺与某个 PyTorch 后端逐位一致。
x32 = x.float()
mean = x32.mean(dim=-1, keepdim=True)
var = ((x32 - mean) ** 2).mean(dim=-1, keepdim=True)
y32 = (x32 - mean) * torch.rsqrt(var + 1e-5)
y = (y32 * weight.float() + bias.float()).to(x.dtype)
```

归约顺序、舍入位置、底层数学函数都可能造成细小差异。因此测试要定义合理容差，同时检查生成 Token；只要求逐位相同会混淆数学错误与浮点实现差异。

## 7. half2：成对搬运不等于纯 half 算术

去 [PagedAttention 的 FP16 分支](../mini_vllm/cuda/paged_attention.cu#L139) 看实际点积：

源码：[mini_vllm/cuda/paged_attention.cu，第 145—152 行](../mini_vllm/cuda/paged_attention.cu#L145)。以下为当前文件的原样摘录。

```cpp
                for (int pair = 0; pair < head_size / 2; ++pair) {
                    const std::size_t offset = cache_offset(
                        physical_block, layer_index, head, page_offset,
                        pair * 2, num_layers, num_heads, head_size);
                    const float2 q_pair = __half22float2(query2[pair]);
                    const float2 k_pair = __half22float2(cache2[offset / 2]);
                    dot += q_pair.x * k_pair.x + q_pair.y * k_pair.y;
                }
```

一次加载两个 half 后，`__half22float2` 得到两个 float，乘加也累积到 float `dot`。这里主要改变访问粒度，帮助处理相邻两个维度。

使用 half2 要同时考虑元素配对与地址对齐。Attention 分支检查 `head_size%2==0`；channels、行跨度、参数偏移等也必须使起始地址按配对布局解释正确。

如果 D 是奇数，不能直接忽略最后一个维度。当前 Attention 使用标量回退处理这种形状。BF16 当前走它自己的标量转换路径，并未因为同样占两字节就自动获得 half2 分支。

## 8. 从容量公式理解显存收益

两份 KV 的总字节数为：

```text
KV_bytes = 2 * pages * layers * heads * page_size * head_dim * bytes_per_element
         = 2 * pages * layers * page_size * channels * bytes_per_element
```

GPT-2 配置 L=12、C=768、page=16，单物理页的 K+V：

| dtype | 单页字节数 | 单页 MiB |
| --- | ---: | ---: |
| FP32 | `2*12*16*768*4 = 1,179,648` | 1.125 |
| FP16 / BF16 | `2*12*16*768*2 = 589,824` | 0.5625 |

MiB 使用 `1024*1024` 字节。任务 11 跨卡传两页 FP16 K+V 时，逻辑 payload 就是 1,179,648 字节。

权重和 KV 的存储大小可以按元素字节比例减半；整张卡总占用还包括 float logits、整数元数据、CUDA runtime、cuBLAS 和 Graph 资源，不保证恰好减半。

## 9. 正确性结果应怎样读

源码：[dev/cuda/test_gpt2_cuda_model_runner.cu，第 230—241 行](../dev/cuda/test_gpt2_cuda_model_runner.cu#L230)。以下为当前文件的原样摘录。

```cpp
    assert(request1->is_finished());
    assert(request2->is_finished());
    assert(request3->is_finished());
    assert(saw_mixed_batch);
    assert(saw_reused_block);
    assert(block_manager.num_free_blocks() == block_manager.num_blocks());
    if (enable_cuda_graph) assert(runner.num_cuda_graphs() > 0);
    assert(global_max_abs_logit_error <
           (data_type == CudaDataType::FP32 ? 0.2 : 3.0));
    if (data_type != CudaDataType::BF16) {
        assert(total_argmax_mismatches == 0);
    }
```

测试对 FP32/FP16 要求 greedy 无不一致；BF16 分支没有这个强制条件。看到程序打印 `passed`，仍要读 `argmax_mismatches` 与误差数值。

本项目已有 BF16 历史实验出现输出不一致，因此当前应把 BF16 视为实验路径，不能在简历写成与 FP32 完全等价、稳定支持所有生成场景。

理解 argmax 对误差的敏感性：假设最大两个 logits 为 1.0000 和 0.9999，只有 0.0001 的间隔。即使整体最大绝对误差很小，也可能交换两者顺序。生成一旦分叉，后续输入前缀也不同，误差不再只是同一输入上的数值比较。

因此定位时用固定相同前缀比较 logits，再讨论 free-running 的完整生成是否一致。不要直接比较已经分叉的两条生成路径并据此定位某一层。

## 10. 调试练习与答案

建议每次只切换一个开关：先 FP32 eager，再 FP16 eager；数值通过后再测试融合和 Graph。否则 dtype、执行顺序与捕获问题同时改变，难以找到第一个分歧。

**题 1：两字节缓冲 `get<__nv_bfloat16>()` 通过检查，能否证明里面是 BF16？**

答案：不能。检查只验证大小，还需追踪初始化写入类型和前向分派。

**题 2：FP16 乘法输入、FP32 累加、FP16 输出，舍入至少发生在哪里？**

答案：FP32 权重转换为 FP16 时，以及 GEMM 结果写回 FP16 时；其他低精度激活算子的写回也可能再次舍入。

**题 3：把最终 logits 改成 float 能恢复前面丢掉的小数位吗？**

答案：不能；它只避免进一步降低输出存储精度。

**题 4：任务 11 中三页 FP16 K+V 的 payload 是多少？**

答案：`3*589824=1,769,472` 字节，即 1.6875 MiB。若经 host 中转，D2H 与 H2D 各搬这么多，总链路字节约为两倍 payload。

**题 5：BF16 指数范围较大，为什么仍不能直接认定比 FP16 更准确？**

答案：数值范围与有效精度是不同维度，模型分布和舍入误差决定实际结果。应根据当前模型的 logits 和生成验证判断，而不是只比较格式名称。

下一篇：[任务 07：融合和 CUDA Graph](task_07_fusion_cuda_graph_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

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
