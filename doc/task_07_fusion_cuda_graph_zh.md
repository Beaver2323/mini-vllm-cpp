# 开发任务 07 学习手册：Residual + LayerNorm 融合与 CUDA Graph

这一篇有两个独立主题：Residual + LayerNorm 融合改变算子边界，CUDA Graph 改变主机提交方式。先分别读懂，再做四组对照，才能知道收益来自哪里。

学习目标：从源码解释融合保留了哪些值、Graph 捕获了什么，以及为什么同一个 N 可能需要多张图。文末已有阶段实验，这里补上完整的推导过程。

## 1. 先建立四种执行模式

| 融合 | Graph | 与基线相比改变了什么 |
| --- | --- | --- |
| 关 | 关 | 普通 kernel/GEMM 逐次提交 |
| 开 | 关 | 减少残差与归一化之间的独立 launch |
| 关 | 开 | 模型算子结构相同，用图重放提交 |
| 开 | 开 | 同时使用两项改变 |

Graph 不会自动帮你把两个 C++ CUDA kernel 合并成一个。融合也不会自动缓存主机发射序列。把两者同时打开后只测一次，无法解释单项效果。

第一遍阅读 [forward](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1079) 时，先沿 `enable_fused_residual_layernorm=false` 分支走，再回到融合分支。

## 2. 调用链与需要打开的符号

```text
Impl::forward
  ├─ 初次 LN1（融合路径仍需要）
  ├─ 每层 Attention projection
  │   └─ fused_residual_layernorm(..., 当前层 LN2 参数)
  ├─ 每层 MLP projection
  │   └─ fused_residual_layernorm(..., 下一层 LN1 或最终 LN 参数)
  └─ 采样行 Gather → LM head → Argmax

同一 forward 外围：
  元数据上传 → Graph 查找 / 捕获 → 执行 / replay → D2H 采样 → 同步
```

| 源码入口 | 关注点 |
| --- | --- |
| [标量融合 kernel](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L389) | 双输出、舍入与归约 |
| [half2 融合 kernel](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L448) | 成对读写与 float 统计 |
| [融合调用包装](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1052) | dtype 与偶数维度分派 |
| [跨层参数选择](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1263) | 当前层结束后用哪一组 LN |
| [Graph 查找](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1105) | key 与动态输入 |
| [图实例化和执行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1321) | capture 结束不等于已经执行 |

## 3. 融合 kernel 为什么仍有两个输出

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 391—419 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L391)。以下为当前文件的原样摘录。

```cpp
template <typename T>
__global__ void residual_layernorm_kernel(
    T* residual_output, T* normalized_output,
    const T* left, const T* right, const T* weight, const T* bias,
    int batch_size, int channels) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batch_size) return;
    __shared__ float reduction[kThreads];

    const std::size_t row_base =
        static_cast<std::size_t>(row) * channels;
    float local_sum = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        const std::size_t index = row_base + channel;
        const T residual = from_float<T>(
            to_float(left[index]) + to_float(right[index]));
        residual_output[index] = residual;
        local_sum += to_float(residual);
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

要同时保存两种数学对象：

```text
residual_output = left + right
normalized_output = LayerNorm(residual_output)
```

后续 Linear 使用 normalized；下一个残差连接需要未归一化的 residual。只输出 normalized 会丢失 skip connection 需要的值。

这也是阅读融合代码的通用方法：先列出所有消费者，再决定哪些中间值可以消失。算子边界消失，不代表每个中间 Tensor 的语义都能删除。

当前实现把 residual 写到 global memory，后面的方差与归一化还会读取它。因此可确定的是减少独立 launch 并合并一部分处理；不能说融合后完全消除了 residual 的显存读写。

## 4. 最关键的两行：先舍入，再归约

```cpp
// 取自上面摘录，单独强调数值边界。
const T residual = from_float<T>(to_float(left[index]) + to_float(right[index]));
local_sum += to_float(residual);
```

未融合 FP16 路径先把加法结果写成 half，后一个 LN 再把 half 读成 float。融合为了尽量保持同样边界，也先转成 T，再参与统计。

教学反例，设某个相加结果为 `1.0003`。FP16 在 1 附近不一定能准确保存它，舍入后可能变为 1。若融合直接把未舍入的 1.0003 累加到 mean，统计量便与未融合路径不同。

PyTorch 对照：

```python
# 展示当前融合想维持的边界。
r = (left.float() + right.float()).to(left.dtype)
x = r.float()
mean = x.mean(-1, keepdim=True)
var = ((x - mean) ** 2).mean(-1, keepdim=True)
y = ((x - mean) * torch.rsqrt(var + 1e-5)
     * weight.float() + bias.float()).to(left.dtype)
```

数学公式相同不保证浮点操作序列相同。读训练/推理优化代码时，既要检查数据依赖，也要检查 cast 的位置。

## 5. 跨层融合：为什么用下一层 LN1

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1263—1276 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1263)。以下为当前文件的原样摘录。

```cpp
            if (config_.enable_fused_residual_layernorm) {
                const bool has_next_layer = layer + 1 < config_.num_layers;
                const T* norm_weight = has_next_layer
                    ? parameters_view.ln1w +
                        static_cast<std::size_t>(layer + 1) * channels
                    : parameters_view.lnfw;
                const T* norm_bias = has_next_layer
                    ? parameters_view.ln1b +
                        static_cast<std::size_t>(layer + 1) * channels
                    : parameters_view.lnfb;
                fused_residual_layernorm(
                    residual_a_.get<T>(), normalized_.get<T>(),
                    residual_b_.get<T>(), projected_.get<T>(),
                    norm_weight, norm_bias, batch_size, channels);
```

某层 MLP 残差输出就是下一层输入，所以下一次消费它的归一化是**下一层 LN1**。最后一层之后没有下一层，应该用最终 `lnfw/lnfb`。

以两层模型手画：

```text
初始 x → LN1[0] → Attention[0]
   → Add + LN2[0] → MLP[0]
   → Add + LN1[1] → Attention[1]
   → Add + LN2[1] → MLP[1]
   → Add + LN_final → LM head
```

若错误使用当前层 LN1[0]，形状完全合法，CUDA 不会报非法访存，但模型语义已经改变。因此验证不能只有“kernel 没崩溃”；需要数值 reference。

初始 LN1 不能一并删掉：它前面没有上一层 MLP 残差可供融合。查看 [初始 LN1 调用](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1145)，理解边界层通常需要单独处理。

## 6. CUDA Graph 保存的是执行计划，不是旧请求的答案

一次普通前向涉及很多 host API 调用。CUDA Graph 记录设备执行节点及依赖，实例化后可复用这套计划，减少后续逐个提交的开销。

当前代码把动态 Token ID、position、页表、采样行号写进地址固定的 device buffer；图中 kernel 读取这些缓冲的**新内容**。

| 项目 | 在当前 Runner 的 replay 间是否可变化 |
| --- | --- |
| Token ID、position、context、物理页号 | 可以，图外更新缓冲内容 |
| 采样行索引的具体数值 | 可以，图外更新 sample_rows |
| 本轮 N、采样行数 R | 对同一图固定，变化时查另一张图 |
| 参数指针、激活缓冲地址、dtype | 当前 Runner 内固定 |
| 模型层数、channels、融合配置 | 当前 Runner 内固定 |

这不是把 PyTorch FX 图序列化，也不是 TorchDynamo 的 Python guard 系统。对于你熟悉的编译器背景，可以类比“复用执行计划”，但不要把两个机制的缓存键和失效条件混为一谈。

## 7. Graph key 为什么是 (N,R)

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1105—1129 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1105)。以下为当前文件的原样摘录。

```cpp
        // Grid/GEMM 随输入行数和采样行数变化；行索引本身在图外更新。
        const auto graph_key = std::make_pair(batch_size, num_logit_rows);
        bool replay_existing_graph = false;
        bool capture_new_graph = false;
        if (config_.enable_cuda_graph) {
            const auto graph = cuda_graphs_.find(graph_key);
            if (graph != cuda_graphs_.end()) {
                check_cuda(
                    cudaGraphLaunch(graph->second.executable, stream_.get()),
                    "launch cached CUDA graph");
                replay_existing_graph = true;
            } else {
                // Stream capture cannot begin behind uncaptured metadata copies.
                // The first use of each Token Batch therefore synchronizes once;
                // subsequent replays keep metadata and graph launch ordered on
                // the same stream without this synchronization.
                check_cuda(
                    cudaStreamSynchronize(stream_.get()),
                    "prepare CUDA graph capture");
                check_cuda(
                    cudaStreamBeginCapture(
                        stream_.get(), cudaStreamCaptureModeThreadLocal),
                    "begin CUDA graph capture");
                capture_new_graph = true;
            }
```

N 决定主体模型的 GEMM 与 grid，R 决定 Gather、LM head 和 Argmax 的规模。只按 N 缓存，会在相同输入行数、不同采样资格时误用输出路径。

例子：

| 本轮 | N | R | 说明 | Graph 行为 |
| --- | ---: | ---: | --- | --- |
| 1 | 4 | 0 | 长 Prompt 的中间 chunk | 新建 `(4,0)` |
| 2 | 4 | 1 | 单请求 Prompt 完成 | 新建 `(4,1)` |
| 3 | 4 | 2 | 两个请求均完成输入 | 新建 `(4,2)` |
| 4 | 4 | 1 | 仅第一个请求完成，行号变了 | 复用 `(4,1)` |
| 5 | 4 | 0 | 再遇中间 chunk | 复用 `(4,0)` |

第 4 轮说明：key 不需要包含具体采样行号，因为行号是 device buffer 中的数据。任务 09 的测试专门覆盖这个场景。

当前按精确 `(N,R)` 存图，没有自动分桶、填充或缓存淘汰。shape 多样时图数量和首次捕获成本可能增长，不能宣称任意动态形状都没有额外开销。

## 8. Capture、Instantiate、Launch 的三个阶段

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1321—1340 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1321)。以下为当前文件的原样摘录。

```cpp
        if (capture_new_graph) {
            CudaGraphEntry entry;
            check_cuda(
                cudaStreamEndCapture(stream_.get(), &entry.graph),
                "end CUDA graph capture");
            check_cuda(
                cudaGraphInstantiate(
                    &entry.executable, entry.graph, nullptr, nullptr, 0),
                "instantiate CUDA graph");
            const auto inserted =
                cuda_graphs_.emplace(graph_key, entry);
            if (!inserted.second) {
                cudaGraphExecDestroy(entry.executable);
                cudaGraphDestroy(entry.graph);
                throw std::logic_error("duplicate CUDA graph batch key");
            }
            check_cuda(
                cudaGraphLaunch(entry.executable, stream_.get()),
                "launch newly captured CUDA graph");
        }
```

第一次形状出现时：

1. 元数据已经上传，先同步，再开始 capture。
2. 调用普通前向里的 kernel/cuBLAS API，把操作记录到图。
3. EndCapture 得到图描述。
4. Instantiate 得到可执行实例。
5. Launch 才让这次输入按图真正执行。

如果省掉第 5 步，第一次遇到形状时采样缓冲可能还是旧值。仅“成功捕获”不代表当前推理已经完成。

后续 replay 不再重新遍历模型的 launch 代码，但仍需要上传新元数据，下载采样 ID，并在返回 CPU 前等待完成。

## 9. 资源生命周期怎样保证图不读悬空地址

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 729—737 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L729)。以下为当前文件的原样摘录。

```cpp
    ~Impl() {
        for (auto& item : cuda_graphs_) {
            if (item.second.executable != nullptr) {
                cudaGraphExecDestroy(item.second.executable);
            }
            if (item.second.graph != nullptr) {
                cudaGraphDestroy(item.second.graph);
            }
        }
```

Graph 中持有设备地址，因此 Runner 不能在图仍可 replay 时把激活缓冲搬到另一个地址。当前缓冲在构造时按容量分配，析构先销毁图资源，再由成员析构释放设备内存。

必须区分：host vector 每轮可以重新分配，因为常规路径上传完成后图读取的是 device buffer；device buffer 的地址则要稳定。

这段析构展示正常资源管理，不代表所有 capture 失败路径都具备完整自动恢复。出现 CUDA 错误后，不应未经验证就把同一个 Runner 当作可继续服务的实例。

## 10. 怎样判断优化到底有没有价值

先用同一配置运行四组合，再查看原始点和 profiler：

- 融合主要观察 kernel 数、kernel 本身时长、显存读写与整个 step 时间。
- Graph 主要观察 host launch 间隙和稳态总时间，同时记录图缓存是否已预热。
- 模型正确性以 logits/Token 对照验证，不能用时间变化代替。
- 单项改善可能被新增 Gather、内存瓶颈或噪声抵消，所以不能承诺组合收益是两者相加。

首次捕获、预热后的 replay、进程总时间是三种指标。文末历史实验说明具体计时方式；再次运行时应确认哪些形状已经进入缓存。

## 11. 练习与参考答案

**题 1：融合只输出 normalized，为什么会错？**

答案：下一次残差连接仍要使用未归一化的 residual。丢掉它会改变 Transformer 公式。

**题 2：把 residual 的 cast 放到 LN 完成后，是否等价？**

答案：浮点上不一定等价，mean/variance 使用的输入改变了。当前代码显式保留先写低精度 residual 的边界。

**题 3：N=4,R=1，采样行从 3 变为 1，要新图吗？**

答案：当前实现不需要；更新 `sample_rows` 内容后可复用 `(4,1)`。若 R 从 1 变为 2，则需要另一张图。

**题 4：Graph 能让 Attention 少读取一半 KV 吗？**

答案：当前 Graph 只改变提交方式，算子读写内容由相同 kernel 决定。KV 工作量不会因 replay 自动减少。

**题 5：已 capture 的模型缓冲可以重新 cudaMalloc 后接着 replay 吗？**

答案：不能直接假设安全；图还记录旧指针。当前通过固定分配避免这个问题，重新分配需要更新或重建执行计划。

下一篇：[任务 08：Prefix Cache 的状态与所有权](task_08_prefix_cache_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[四种图与缓存的区别](from_pytorch/04_pytorch_to_cuda.md#7-四种容易混为一谈的缓存图)。
如果你熟悉 FX/Inductor，要特别区分编译图和 CUDA Graph 的运行时重放。

当前代码已加入任务 09 的采样行裁剪，图缓存键已更新；下方任务 07 性能表保留原始实验口径。

这份文档用于沿着真实代码学习。建议打开编辑器后左右分屏：左边放本文，右边按每节给出的
文件和函数跳转。行号对应当前提交；以后代码变化时优先按函数名搜索。

## 1. 学习目标与代码导航

学完后应能回答五个问题：

1. GPT-2 Pre-LN Block 中，哪些 Residual 与 LayerNorm 可以融合？
2. 为什么融合路径仍保留一次初始 LayerNorm？
3. CUDA Graph 固定了什么，为什么每轮仍能换 Token 和页表？
4. 为什么 Graph Cache 需要同时考虑 Packed Token 数和采样行数？
5. 为什么 Kernel 数下降不一定带来端到端加速？

| 学习点 | 先看哪里 | 再看调用点 |
| --- | --- | --- |
| 功能开关 | [`GPT2CudaConfig`](../mini_vllm/cuda/gpt2_cuda_model_runner.cuh#L19-L29) | [`benchmark` 参数解析](../benchmark/benchmark_gpt2_cuda_serving.cu#L190-L230) |
| 通用融合 Kernel | [`residual_layernorm_kernel`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L389-L446) | [`fused_residual_layernorm`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1052-L1076) |
| FP16 `half2` Kernel | [`residual_layernorm_half2_kernel`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L448-L514) | `channels % 2 == 0` 分发分支 |
| Attention 后融合 | [`forward` 中的第一次融合](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1191-L1234) | Attention Projection 之后 |
| MLP 后融合 | [`forward` 中的第二次融合](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1255-L1301) | 下一层 LN1 或最终 LN |
| Graph 元数据边界 | [`forward` 的 H2D 拷贝](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1079-L1099) | Capture/Replay 判断之前 |
| Graph Capture/Replay | [`forward` 的 Graph 分支](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1107-L1130) | 计算区及 `cudaGraphLaunch` |
| Graph 缓存结构 | [`CudaGraphEntry`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1407-L1411) | 析构释放见 700--709 行 |
| 正确性验证 | [`test_gpt2_cuda_model_runner.cu`](../dev/cuda/test_gpt2_cuda_model_runner.cu#L117-L250) | `validate_and_commit` |
| 性能测量 | [`run_once`](../benchmark/benchmark_gpt2_cuda_serving.cu#L124-L180) | `main` 中 Warmup/Repeat 复用 Engine |

## 2. 先从完整调用链进入

不要先从 CUDA Kernel 第一行开始读。先看一次请求怎样到达它：

```text
benchmark::run_once
  └─ GPT2CudaEngine::step
       ├─ Scheduler::schedule
       ├─ GPT2CudaModelRunner::run
       │    ├─ prepare_packed_model_input
       │    └─ Impl::forward<T>
       │         ├─ H2D 动态元数据
       │         ├─ Graph Replay，或普通/捕获执行
       │         ├─ Residual + LayerNorm
       │         └─ Logits + Argmax
       └─ Scheduler::commit
```

入口在
[`GPT2CudaEngine::step`](../mini_vllm/cuda/gpt2_cuda_engine.hpp#L75-L98)：

```cpp
SchedulerOutput output = scheduler_.schedule();
result.sampled_token_ids = model_runner_.run(output);
scheduler_.commit(output, result.sampled_token_ids);
```

这三行划分了控制面和执行面：Scheduler 决定本轮算哪些 Token；Runner 只消费已经整理好的
调度结果；Commit 在 GPU 计算完成后更新 Sequence 状态。Fusion 和 CUDA Graph 都只修改
Runner 的执行面，不改变 Scheduler 语义。

Runner 的上层入口在
[`GPT2CudaModelRunner::Impl::run`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L740-L798)：

```cpp
ModelInput input = prepare_packed_model_input(
    output, block_manager_, max_context_length_,
    max_blocks_per_sequence_, num_pages_);
const std::vector<int> token_samples =
    config_.data_type == CudaDataType::FP16
        ? forward<__half>(input)
        : /* BF16 或 FP32 */;
```

`run` 先把多个请求压成一个 Packed Token Batch，再按精度实例化 `forward<T>`。因此后文的
`batch_size` 指本轮总 Token 数，不等于请求数。

## 3. Residual + LayerNorm 为什么能融合

### 3.1 先认清 GPT-2 的 Pre-LN 数据流

一层 Transformer 可以简化为：

```text
residual_a
   ├─ LN1 → Attention → Projection ─┐
   └─────────────────────────────────┴─ Add → residual_b
                                        └─ LN2 → MLP ─┐
   residual_b ─────────────────────────────────────────┴─ Add → residual_a(next)
                                                           └─ next LN1 / final LN
```

未融合时，每个 `Add` 和后面的 `LayerNorm` 各启动一个 Kernel。二者之间只有一个中间
`residual_*` Tensor，没有其他消费者修改它，所以可以在一次 Launch 中完成：

```text
left + right → residual_output → 求均值/方差 → normalized_output
```

仍要同时写出两个结果：`residual_output` 是后续残差支路的输入，`normalized_output` 是
Attention 或 MLP 的输入。融合不是把残差结果删掉。

### 3.2 通用 Kernel：一行 Token 对应一个 CUDA Block

实现位置：
[`residual_layernorm_kernel`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L392-L446)。

```cpp
const int row = blockIdx.x;       // 第几个 Packed Token
const int thread = threadIdx.x;   // 负责该 Token 的哪些 channel
__shared__ float reduction[kThreads];

for (int channel = thread; channel < channels; channel += blockDim.x) {
    const std::size_t index = row_base + channel;
    const T residual = from_float<T>(
        to_float(left[index]) + to_float(right[index]));
    residual_output[index] = residual;
    local_sum += to_float(residual);
}
```

Grid 是 `batch_size`，所以一个 Block 独占一行 `[channels]`。线程以步长
`blockDim.x` 遍历 Channel，然后用共享内存树形归约求均值。接下来再次遍历
`residual_output` 求方差：

```cpp
const float shifted =
    to_float(residual_output[row_base + channel]) - mean;
local_variance += shifted * shifted;
```

这里读取刚写入的 `residual_output`，而不是直接使用寄存器中的 `left + right`。原因是低精度
Eager 基线先把 Add 结果舍入为 FP16/BF16，再由 LayerNorm 转回 FP32 归约。融合 Kernel
也保留这个舍入边界，避免 Fusion 开关引入额外数值差异。

最后一遍完成仿射归一化：

```cpp
normalized_output[index] = from_float<T>(
    (to_float(residual_output[index]) - mean) * inverse_stddev *
        to_float(weight[channel]) +
    to_float(bias[channel]));
```

学习时重点检查三个同步点：Residual 写完以后、均值归约以后、方差归约以后。删除其中任何
一个都可能让同一 Block 内线程读到未完成的数据。

### 3.3 FP16 `half2` 路径

实现位置：
[`residual_layernorm_half2_kernel`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L448-L514)。

```cpp
const int pair_width = channels / 2;
__half2* residual2 = reinterpret_cast<__half2*>(residual_output);
const __half2* left2 = reinterpret_cast<const __half2*>(left);
const __half2* right2 = reinterpret_cast<const __half2*>(right);

const __half2 residual = __hadd2(left2[index], right2[index]);
residual2[index] = residual;
const float2 values = __half22float2(residual);
local_sum += values.x + values.y;
```

一个线程一次处理两个相邻 FP16 Channel。`half2` 减少指令和访存事务，但要求 Channel 数
是偶数。分发函数明确保留标量回退：

```cpp
if constexpr (std::is_same<T, __half>::value) {
    if (channels % 2 == 0) {
        residual_layernorm_half2_kernel<<<...>>>(...);
    } else {
        residual_layernorm_kernel<T><<<...>>>(...);
    }
}
```

调用点在
[`fused_residual_layernorm`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1052-L1076)。
GPT-2 124M 的 `channels=768`，所以实际会进入 `half2`。

### 3.4 两处融合调用为什么使用不同 LayerNorm 参数

Attention Projection 后的调用位于
[`forward` 1053--1061 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1198-L1206)：

```cpp
fused_residual_layernorm(
    residual_b, normalized,
    residual_a, projected,
    parameters_view.ln2w + layer * channels,
    parameters_view.ln2b + layer * channels,
    batch_size, channels);
```

它计算 `residual_b = residual_a + attention_projection`，紧接着产生本层 MLP 所需的
`LN2(residual_b)`。

MLP Projection 后的调用位于
[`forward` 1118--1131 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1263-L1276)：

```cpp
const bool has_next_layer = layer + 1 < config_.num_layers;
const T* norm_weight = has_next_layer
    ? parameters_view.ln1w + (layer + 1) * channels
    : parameters_view.lnfw;

fused_residual_layernorm(
    residual_a, normalized,
    residual_b, projected,
    norm_weight, norm_bias, batch_size, channels);
```

中间层要提前算下一层的 LN1；最后一层没有下一层 LN1，因此改用模型最终的 `ln_f`。
这也解释了为什么开启融合后，在进入第 0 层之前仍要单独执行一次初始 LN1：它前面没有
可合并的 MLP Residual。代码在 1000--1005 行。

### 3.5 Kernel 数怎样算

12 层模型未融合时：每层 2 个 Residual + 2 个层内 LayerNorm，最后再加 1 个 Final LN，
共 `24 Residual + 25 LayerNorm = 49` 次 Launch。融合后：第 0 层初始 LN1 为 1 次，
每层两次融合为 24 次，共 25 次，单次模型前向减少 24 次 Launch。

Nsight 记录 10 次模型前向，因此模型计算 Kernel 从 2080 降到 1840：

```text
2080 - 1840 = 240 = 10 × 24
```

## 4. CUDA Graph：固定地址，更新数据

### 4.1 为什么当前 Runner 适合 Capture

Runner 构造时一次性分配最大容量的设备 Buffer。实现位于
[`Impl` 构造函数](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L676-L726)：

```cpp
token_ids_(max_num_tokens_),
positions_(max_num_tokens_),
context_lengths_(max_num_tokens_),
slot_mapping_(max_num_tokens_),
block_tables_(max_num_tokens_ * max_blocks_per_sequence_),
key_cache_(cache_elements(), storage_size()),
residual_a_(batch_channels(), storage_size()),
/* 其余中间 Tensor */
```

这些 Buffer 在 Engine 生命周期内地址不变。Graph 捕获的是 Kernel 拓扑、参数和设备指针；
指针指向的内容可以在 Replay 前更新。这是理解本实现的核心。

### 4.2 动态元数据为什么放在 Graph 外

`forward<T>` 一开始先异步上传五组动态数据，位置在 952--960 行：

```cpp
copy_metadata(token_ids_, input.token_ids, "copy token ids");
copy_metadata(positions_, input.positions, "copy positions");
copy_metadata(context_lengths_, input.context_lengths, "copy context lengths");
copy_metadata(slot_mapping_, input.slot_mapping, "copy slot mapping");
copy_metadata(block_tables_, input.block_tables, "copy block tables");
```

`copy_metadata` 使用同一条 Stream 的 `cudaMemcpyAsync`，实现在 1201--1214 行。随后才调用
`cudaGraphLaunch`。同一 Stream 保证顺序：Graph 内 Kernel 开始读之前，新的 Token、位置、
上下文长度和页表已经写到固定设备地址。

```text
同一 CUDA Stream：
H2D(metadata round N) → cudaGraphLaunch(graph shape K) → D2H(sample)
```

如果把 H2D Capture 进图，源 Host 指针和拷贝长度也会成为 Graph Node 参数，当前教学实现
需要额外做 pinned staging buffer 或 Graph Node Update。把动态拷贝放在图外，边界更清楚。

### 4.3 Cache Key 为什么是 `(batch_size, num_logit_rows)`

查找点在 964--966 行：

```cpp
const auto graph_key = std::make_pair(batch_size, num_logit_rows);
const auto graph = cuda_graphs_.find(graph_key);
```

这里的 `batch_size = input.batch_size()`，即 Packed Token 数。它会改变 Elementwise Kernel
Grid、LayerNorm/Fusion Grid 和模型主干 GEMM 维度。任务 09 加入采样行裁剪后，
`num_logit_rows` 独立决定 LM Head、Argmax 和 logits 行数，因此一起进入 Key。
具体采样行索引作为图外更新的动态数据，详见 [任务 09](task_09_sample_rows_zh.md)。

模型层数、Channels、精度、Fusion 开关和 Buffer 地址在同一个 Runner 构造后不变，所以
当前无需进入 Key。若以后让同一 Runner 动态切换精度、模型或 Fusion，Key 也必须扩展。

### 4.4 首次 Capture 与后续 Replay

实现位置：
[`forward` 962--985、1166--1185 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1107-L1130)。

首次出现形状：

```cpp
cudaStreamSynchronize(stream);          // 等图外 H2D 完成
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
// 执行区：Embedding → 12 Blocks → Logits → Argmax
cudaStreamEndCapture(stream, &entry.graph);
cudaGraphInstantiate(&entry.executable, entry.graph, nullptr, nullptr, 0);
cuda_graphs_.emplace(graph_key, entry);
cudaGraphLaunch(entry.executable, stream); // 真正执行这次请求
```

`BeginCapture` 前同步是因为同一 Stream 前面已有未捕获的元数据拷贝。Capture 记录执行区，
结束并实例化后还要 Launch 一次，才能为当前请求产生输出。

再次遇到相同形状：

```cpp
cudaGraphLaunch(graph->second.executable, stream_.get());
replay_existing_graph = true;
```

`if (!replay_existing_graph)` 会跳过逐个 Kernel 的 Host 提交。最后的 Sample D2H 和 Stream
同步仍在 Graph 外，因为 CPU Scheduler 需要本轮生成 Token 才能 Commit。

### 4.5 Graph 资源生命周期

缓存定义在 1251--1255 行：

```cpp
struct CudaGraphEntry {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
};
std::map<std::pair<int, int>, CudaGraphEntry> cuda_graphs_;
```

Runner 析构函数在 700--709 行先销毁 `cudaGraphExec_t`，再销毁 `cudaGraph_t`。这是资源
所有权的一部分，不能只缓存句柄而不释放。

## 5. Benchmark 调用点与四组 A/B

参数从
[`parse_options`](../benchmark/benchmark_gpt2_cuda_serving.cu#L190-L230)
进入 `GPT2CudaConfig`：

```cpp
} else if (argument == "--fusion") {
    options.enable_fusion = true;
} else if (argument == "--cuda-graph") {
    options.enable_cuda_graph = true;
}
```

Benchmark 必须复用同一个 Engine。正确调用点位于
[`main` 402--430 行](../benchmark/benchmark_gpt2_cuda_serving.cu#L402-L430)：

```cpp
GPT2CudaEngine engine(...);
const RunResult warmup = run_once(engine, 1000);
for (int repeat = 0; repeat < options.repeats; ++repeat) {
    RunResult run = run_once(engine, 2000 + repeat * 10);
}
```

Warmup 建立 Graph Cache，正式 Repeat 才测稳定 Replay。若每次 `run_once` 前重新构造
Engine，Graph Cache 也会重建，测到的是 Capture 成本。

RTX 3090、FP16、4 请求、Prompt 8/16/24/32、每请求输出 4 Token、Token Budget 64：

| Fusion | CUDA Graph | 总时间中位数 | 吞吐中位数 | 相对基线 |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 关闭 | 6.044 ms | 2647.036 tok/s | 基线 |
| 开启 | 关闭 | 7.599 ms | 2105.604 tok/s | -20.5% |
| 关闭 | 开启 | 5.215 ms | 3068.092 tok/s | +15.9% |
| 开启 | 开启 | 5.016 ms | 3189.767 tok/s | +20.5% |

原始数据位于 `benchmark/results/gpt2_cuda_task07_*_rtx3090.json`。融合 Eager 变慢说明减少
Launch 只是手段。当前融合 Kernel 需要多轮 Block 同步、共享内存归约，并重复读取
`residual_output`；节省的 Launch 成本小于增加的 Kernel 执行成本。因此配置默认值保持
`false`，见 `GPT2CudaConfig` 27 行。

## 6. 正确性测试应沿哪里看

测试入口是
[`dev/cuda/test_gpt2_cuda_model_runner.cu`](../dev/cuda/test_gpt2_cuda_model_runner.cu#L117-L250)。

建议按下面顺序下断点：

1. `main:196`：`scheduler.schedule()`，观察 `output.num_batched_tokens`。
2. `validate_and_commit`：比较 CPU 完整前缀 logits 与 GPU logits。
3. `GPT2CudaModelRunner::Impl::run:730`：检查 Packed ModelInput。
4. `forward<T>:952`：检查本轮 H2D 元数据。
5. `forward<T>:965`：观察 Graph 第一次 Miss、后续 Hit。
6. `forward<T>:1054/1128`：观察两种 Fusion 调用。
7. `main:227`：确认 Graph Cache 非空。

测试覆盖混合 Prefill/Decode、跨 16 Token 页边界和 Block 释放复用。验收结果：

```text
max_abs_logit_error=0.117905
argmax_mismatches=0
graph_cache_size=3
Compute Sanitizer memcheck=0 errors
```

`max_abs_logit_error` 不为 0 是 FP16 与 CPU FP32 的数值差异；服务最终输出还要检查
Argmax Token 一致。这里记录的是任务 07 当时的历史结果。任务 09 后，Graph Key 变为两种行数的组合，
当前相同调度测试可能得到不同的 Graph 数，应以当前测试输出为准。

## 7. 复现与调试命令

```bash
cd /home/users/zyf/zyf_llm.c/llm.c

make GPU_COMPUTE_CAPABILITY=86 \
  test_gpt2_cuda_model_runner benchmark_gpt2_cuda_serving

CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_model_runner \
  --precision fp16 --cuda-graph

CUDA_VISIBLE_DEVICES=0 ./benchmark_gpt2_cuda_serving \
  --precision fp16 --fusion --cuda-graph \
  --token-budget 64 --repeats 5 \
  --json /tmp/task07.json --csv /tmp/task07.csv
```

要观察 Graph Node 和 Kernel 数：

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,cublas --cuda-graph-trace=node \
  -o /tmp/task07_graph \
  ./benchmark_gpt2_cuda_serving \
  --precision fp16 --fusion --cuda-graph --token-budget 64 --repeats 5
```

## 8. 建议亲手完成的四个练习

1. 在 `forward<T>` 的 Graph 查找前打印 `batch_size`，记录一次负载产生哪些 Graph Key。
2. 临时关闭 `half2` 分发，只走通用 FP16 Kernel，比较 logits 和融合 Eager 时间。
3. 把 Warmup 的 Engine 和 Repeat 的 Engine 分开构造，观察 Graph 性能为何退化。
4. 用 Nsight 对比未融合/融合的 Kernel 次数，再对比 Eager/Graph 的 CUDA API 次数。

完成练习后，你应能解释：Fusion 减少设备 Kernel 数；CUDA Graph 减少 CPU 对这些 Kernel
的逐个提交。Graph 依赖稳定设备地址，动态 Token 和页表在 Replay 前写入这些地址。实验中
Graph 是主要收益，融合单独使用反而变慢，所以结论来自四组 A/B，而不是来自 Kernel 数推断。
