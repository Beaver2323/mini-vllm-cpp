# 开发任务 04：将 CUDA PagedAttention 接入 GPU ModelRunner

任务 03 只验证 Attention；这一篇把 Embedding、LayerNorm、矩阵乘、Attention、MLP 和采样连成完整 GPU 前向。阅读时先关闭脑中的融合与 CUDA Graph，把普通执行路径走通。

学习目标：能够画出一个 Transformer 层的缓冲区读写关系，解释为什么每步不需要把权重、KV 和全部 logits 搬回 CPU。

## 1. 先找到三个不同层次的入口

| 层次 | 源码 | 职责 |
| --- | --- | --- |
| 服务驱动 | [gpt2_cuda_engine.hpp](../mini_vllm/cuda/gpt2_cuda_engine.hpp) | 接收请求，调用调度与提交 |
| Runner 接口 | [gpt2_cuda_model_runner.cuh](../mini_vllm/cuda/gpt2_cuda_model_runner.cuh) | 对外提供 run、统计与调试接口 |
| 数值执行 | [gpt2_cuda_model_runner.cu](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1079) | 管理设备缓冲、cuBLAS、kernel 与同步 |

```text
GPT2CudaEngine::step
  ├─ scheduler.schedule
  ├─ GPT2CudaModelRunner::run
  │   └─ Impl::run
  │       ├─ prepare_packed_model_input
  │       ├─ 选择 forward<float / half / bfloat16>
  │       └─ 把采样结果映射回请求
  └─ scheduler.commit
```

`Impl` 是私有实现，帮助头文件隐藏 CUDA 细节。它不是额外的网络服务，也不是另一个进程。当前 Runner 在一个进程内管理某张卡上的资源。

任务 04 初期曾使用单 Token GPU 微步；**当前源码已经包含任务 05 的 packed 路径**。文末历史数字保留当时配置，下面以当前代码为准。

## 2. 从 torch.empty 到 DeviceBuffer

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 70—94 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L70)。以下为当前文件的原样摘录。

```cpp
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count == 0) throw std::invalid_argument("zero-sized CUDA buffer");
        check_cuda(
            cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};
```

这段类承担了你在 PyTorch Tensor 中习惯由框架处理的最基础职责：申请设备内存、保存指针与大小、析构时释放。

逐行抓住三个限制：

1. `count` 是元素数，实际申请 `count*sizeof(T)` 字节。
2. 禁止复制，避免两个对象持有同一个裸指针后重复 `cudaFree`。
3. `get()` 只返回指针，不携带 Tensor 的 shape/stride/device 元信息；这些约束由上层保存和检查。

不要将它理解成拥有完整 PyTorch Tensor 功能的替代品。没有自动广播、引用视图、autograd 或跨设备复制语义。

当前代码还用 `DeviceTensorBuffer` 管理按运行配置选择的 2/4 字节存储，具体精度边界见任务 06。

## 3. stream 与 cuBLAS 必须处在同一执行序列

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 139—165 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L139)。以下为当前文件的原样摘录。

```cpp
class CudaStream {
public:
    CudaStream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }
    ~CudaStream() {
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle(cudaStream_t stream, CudaDataType data_type) {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
        check_cublas(
            cublasSetMathMode(
                handle_, data_type != CudaDataType::FP32
                    ? CUBLAS_DEFAULT_MATH
                    : CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode");
```

`cudaStreamNonBlocking` 创建独立执行队列；`cublasSetStream` 把 GEMM 放到同一队列。这样一层中的：

```text
LayerNorm 写 normalized
  → cuBLAS 读 normalized / 写 qkv
  → split 读 qkv
  → Attention 读 Q、K、V
```

无需每个算子后都 `cudaDeviceSynchronize`。依赖由 stream 顺序表达，最后需要 CPU 读取采样结果时才等待。

如果 cuBLAS 留在另一个无显式依赖的 stream，上述指针依赖不会自动由 C++ 语句顺序保证。对于 PyTorch 开发者，这相当于自己承担原本由当前流约定及框架调度维护的执行顺序。

## 4. 缓冲区提前分配：容量与本轮 N 不相等

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 698—715 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L698)。以下为当前文件的原样摘录。

```cpp
          key_cache_(cache_elements(), storage_size()),
          value_cache_(cache_elements(), storage_size()),
          residual_a_(batch_channels(), storage_size()),
          residual_b_(batch_channels(), storage_size()),
          normalized_(batch_channels(), storage_size()),
          qkv_(static_cast<std::size_t>(max_num_tokens_) *
               3 * config_.channels, storage_size()),
          query_(batch_channels(), storage_size()),
          key_(batch_channels(), storage_size()),
          value_(batch_channels(), storage_size()),
          attention_(batch_channels(), storage_size()),
          projected_(batch_channels(), storage_size()),
          hidden_(static_cast<std::size_t>(max_num_tokens_) *
                  4 * config_.channels, storage_size()),
          sampled_hidden_(static_cast<std::size_t>(max_logit_rows()) *
                          config_.channels, storage_size()),
          logits_(static_cast<std::size_t>(max_logit_rows()) *
                  config_.padded_vocab_size) {
```

这里按最大调度 Token 数分配激活，但每轮只使用前 N 行。`max_num_sequences` 与 `max_num_tokens` 是不同容量：一个长 Prefill 请求可以独占很多 Token 行。

| 缓冲 | 当前有效形状 | 覆盖时机 |
| --- | --- | --- |
| `residual_a/b` | `[N,C]` | 每个残差分支轮流更新 |
| `normalized` | `[N,C]` | LN1、LN2、最终 LN 重复使用 |
| `qkv` | `[N,3C]` | 每层 QKV 投影后 |
| `query/key/value` | 各 `[N,H,D]` | 每层 split 后 |
| `attention/projected` | `[N,C]` | Attention 与投影后 |
| `hidden` | `[N,4C]` | MLP 中间层 |
| `logits` | `[R,Vp]` | 仅当前需要采样的行，详见任务 09 |

这些中间激活没有按 `L` 再分配一份，因为推理不需要保存每一层激活供 backward 使用。KV 则包含层维度，下一轮仍要读取每一层的历史 K/V，不能在层间覆盖同一份。

## 5. H2D 元数据与 D2H 输出

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1091—1104 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1091)。以下为当前文件的原样摘录。

```cpp
        copy_metadata(token_ids_, input.token_ids, "copy token ids");
        copy_metadata(positions_, input.positions, "copy positions");
        copy_metadata(
            context_lengths_, input.context_lengths,
            "copy context lengths");
        copy_metadata(
            slot_mapping_, input.slot_mapping, "copy slot mapping");
        copy_metadata(
            block_tables_, input.block_tables, "copy block tables");

        const int num_logit_rows = static_cast<int>(last_logit_token_indices_.size());
        if (config_.enable_sample_row_pruning && num_logit_rows > 0) {
            copy_metadata(sample_rows_, last_logit_token_indices_, "copy sample rows");
        }
```

每步复制的是小型整数数组。权重在初始化时已经上传，KV 也一直位于 GPU；不会每轮把完整历史 K/V 经 CPU 往返一次。

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1355—1368 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1355)。以下为当前文件的原样摘录。

```cpp
    void copy_metadata(
        DeviceBuffer<int>& destination, const std::vector<int>& source,
        const char* operation) {
        if (source.size() > destination.count()) {
            throw std::out_of_range("CUDA metadata exceeds buffer capacity");
        }
        const std::size_t bytes = source.size() * sizeof(int);
        check_cuda(
            cudaMemcpyAsync(
                destination.get(), source.data(), bytes,
                cudaMemcpyHostToDevice, stream_.get()),
            operation);
        last_host_to_device_bytes_ += bytes;
    }
```

`source` 是 host vector，`destination` 是预分配的设备数组。异步 API 并不保证 pageable host 内存一定与计算重叠；这里首先依赖的是正确的 stream 顺序和函数结束前的同步，不能仅看到 `Async` 就宣称消除了传输开销。

以 `N=4,max_blocks=4,R=2` 为例，基础元数据包含四个 N 长数组和 N×4 页表，共 `(4*4+4*4)*4=128` 字节；开启采样行裁剪还上传 R 个行号，增加 8 字节。这个数不含一次性的权重上传。

## 6. 对照 PyTorch 阅读 QKV 与 Attention 调用点

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1166—1189 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1166)。以下为当前文件的原样摘录。

```cpp
            matmul(
                qkv_.get<T>(), normalized_.get<T>(),
                parameters_view.qkvw +
                    static_cast<std::size_t>(layer) * 3 * channels * channels,
                parameters_view.qkvb +
                    static_cast<std::size_t>(layer) * 3 * channels,
                batch_size, channels, 3 * channels);
            split_qkv_kernel<T><<<
                blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
                qkv_.get<T>(), query_.get<T>(), key_.get<T>(), value_.get<T>(),
                batch_size, channels);
            check_last_kernel("split_qkv_kernel");

            check_cuda(
                paged_attention_decode(
                    query_.get<T>(), key_.get<T>(), value_.get<T>(),
                    key_cache_.get<T>(), value_cache_.get<T>(),
                    block_tables_.get(), context_lengths_.get(),
                    slot_mapping_.get(), attention_.get<T>(), batch_size,
                    num_pages_, config_.num_layers, layer,
                    config_.num_heads, channels / config_.num_heads,
                    max_blocks_per_sequence_, max_context_length_,
                    stream_.get()),
                "paged_attention_decode");
```

对应 PyTorch 的概念顺序：

```python
# 教学伪代码，paged_attention 不是本项目提供的 Python API。
qkv = F.linear(normalized, qkv_weight, qkv_bias)  # [N,3C]
q, k, v = qkv.chunk(3, dim=-1)                   # 各 [N,C]
att = paged_attention(q, k, v, kv_pool, metadata)
```

真实 split 把数据写进三份独立缓冲，不能简单把它当成 PyTorch 的零拷贝 view。Attention 使用 `channels/num_heads` 作为 D，缓存地址还需要当前 `layer`。

参数层偏移为 `layer*3*C*C`，因为每层 QKV weight 有 `3C×C` 个元素。bias 偏移为 `layer*3*C`，不能复用权重的 stride。

函数名仍叫 `paged_attention_decode`，但当前接收的是 packed 输入行。代码行为由参数和可见长度决定，不能仅根据历史函数名判断它只支持 Decode。

## 7. cuBLAS 的转置参数为什么看起来反了

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 985—997 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L985)。以下为当前文件的原样摘录。

```cpp
    template <typename T>
    void matmul(
        T* output, const T* input, const T* weight,
        const T* bias, int batch_size, int input_width,
        int output_width) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if constexpr (std::is_same<T, float>::value) {
            check_cublas(cublasSgemm(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, input_width, input, input_width, &beta,
                output, output_width), "cublasSgemm");
```

PyTorch `F.linear(X,W)` 使用行优先理解：`X[N,I]`、`W[O,I]`，结果 `Y=XWᵀ[N,O]`。

传统 cuBLAS 接口按列优先解释同一块内存：

| C++ 中的行优先数组 | cuBLAS 对同一指针的列优先解释 |
| --- | --- |
| X `[N,I]` | Xᵀ `[I,N]` |
| W `[O,I]` | Wᵀ `[I,O]` |
| Y `[N,O]` | Yᵀ `[O,N]` |

因此调用计算 `Yᵀ = W × Xᵀ`：第一个操作数 weight 要 `OP_T`，第二个 input 用 `OP_N`，`m=O,n=N,k=I`。这是布局解释，不是额外启动一次转置 kernel。

`beta=0` 表示覆盖输出，旧缓冲内容不参与结果。bias 由后续 kernel 加入，本接口没有把 bias 融合进 GEMM epilogue。

手算检查：N=2,I=3,O=4 时，输出只有 8 个元素；如果把 `m,n` 填成 2、4 却保持原 leading dimension，就可能写出形状错误但地址仍合法的结果，单纯 memcheck 不一定发现。

## 8. 残差为什么需要保存两条值

未融合路径的一层可以写成：

```python
# 数学对照；实际使用预分配缓冲。
x1 = x + attn(ln1(x))
x2 = x1 + mlp(ln2(x1))
```

`residual_a` 保存 x，Attention 投影写入 `projected`，相加后写到 `residual_b`。后续 LN2 写 `normalized`，MLP 最终再写 `projected`，相加回 `residual_a`。

```text
a(x) ───────┐                    b(x1) ──────┐
LN1→Attn→projected → Add → b      LN2→MLP→projected → Add → a(x2)
```

读写同一个缓冲前要检查旧值是否仍有消费者。因为残差连接还要读原输入，所以不能随意把所有中间结果都原地写入 `residual_a`。

任务 07 的融合会同时输出残差值和归一化值，仍保留两种数学状态；它没有把残差连接删除。

## 9. 函数返回时 CPU 究竟得到了什么

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1342—1352 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1342)。以下为当前文件的原样摘录。

```cpp
        std::vector<int> sampled(num_logit_rows);
        if (num_logit_rows > 0) check_cuda(
            cudaMemcpyAsync(
                sampled.data(), sampled_token_ids_.get(),
                sampled.size() * sizeof(int), cudaMemcpyDeviceToHost,
                stream_.get()),
            "copy sampled token ids");
        check_cuda(
            cudaStreamSynchronize(stream_.get()),
            "finish CUDA GPT-2 micro batch");
        return sampled;
```

GPU argmax 的结果只是 R 个整数，拷回后再转换成与调度 item 等长的数组，未完成 chunk 对应 `-1`。

`cudaStreamSynchronize` 使返回的 host Token 可以立即被 `Scheduler::commit` 使用。这也解释了 Benchmark 围住 `step()` 的主机时间为什么包含 GPU 执行。

调试接口 `last_logits_for_testing()` 额外复制 logits，只应在数值检查时调用。当前日志行数是 R，配套 `last_logit_token_indices()` 才能知道每行属于 packed 输入的哪一行；不要把调试数据误作常规输出传输量。

## 10. 正确性定位与练习

读 [GPU Runner 测试](../dev/cuda/test_gpt2_cuda_model_runner.cu#L200)，它不仅检查最后是否结束，还要求实际出现混合 Prefill/Decode、释放后的页复用，并对 CPU reference 比较 logits/Token。

遇到错误推荐沿着第一个分歧排查：

1. 输入 Token、position、context、页表是否一致。
2. Embedding 输出是否一致。
3. 第一个出错层的 LN、QKV、Attention、MLP 哪一项开始偏离。
4. 精度路径一致时再比较最终 logits；避免仅凭最后 Token 不同就认定 Attention 错误。

**题 1：所有激活按层保存，是否更接近 PyTorch？**

答案：普通训练会为 backward 保留更多状态，但当前仅做推理，层间可复用缓冲。按层保存会增加显存，并不是推理正确性的要求。

**题 2：每次 run 都把完整 logits 拷回 CPU 吗？**

答案：常规路径仅返回采样 ID。测试接口可以额外下载 logits，这部分不应混入正常服务传输统计。

**题 3：`cudaMemcpyAsync` 之后立刻读取 host 输出安全吗？**

答案：只有依赖和完成条件已满足才安全。当前代码通过同一 stream 上的同步保证，然后才返回。

**题 4：GPU 0 的 Runner 能否在切到 GPU 1 后随便析构？**

答案：资源属于创建时的设备。当前外层入口和析构维护设备上下文，任务 11 会解释 DeviceGuard 与跨卡线程的关系。

下一篇：[任务 05：为什么一轮应打包多个 Token](task_05_multi_token_prefill_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[用 PyTorch 经验读 CUDA Runner](from_pytorch/04_pytorch_to_cuda.md)。
下文的单 Token 微批次与内存数值是任务 04 历史基线。当前 GPU 已使用 Packed Prefill；任务 09
后 logits 只返回采样行，H2D 还会上传采样行索引。CPU 路径仍保留单 Token 微批次。

**状态：已完成 FP32 端到端基线。**

## 任务目标

任务 03 只证明了独立 PagedAttention Kernel 正确。任务 04 要回答更接近 vLLM 的问题：
Scheduler 产生的请求元数据怎样进入 GPU；BlockManager 的物理页怎样对应设备 KV Cache；
Attention 前后的 Transformer 层在哪里执行；生成 Token 怎样提交回请求状态。

本任务新增约 1600 行 C++/CUDA，其中 GPU ModelRunner 约 760 行，剩余代码主要是共享
元数据、Engine 封装、模型级测试和服务 Benchmark。

## 端到端架构

```text
Sequence / BlockManager / Scheduler（CPU 控制面）
                    │
                    │ ModelInput
                    │ token / position / context length
                    │ slot mapping / block table
                    ▼
GPT2CudaModelRunner（持久化设备内存）
  ├─ GPT-2 权重
  ├─ K/V Cache [page, layer, head, token, dim]
  ├─ 调度元数据 Buffer
  ├─ 中间激活与 logits
  └─ CUDA Stream + cuBLAS Handle
                    │
                    ▼
Embedding → 12 × Transformer Layer → Final LN → LM Head
                    │
                    ▼
设备 Argmax → 只回传 Token ID → Scheduler::commit
```

CPU 和 CUDA Runner 共用 `model_input.hpp`。这样两条路径对 Token 位置、逻辑页、物理页
和页内 Slot 的解释只有一份，避免两个后端各自复制一套容易漂移的索引代码。

## ModelInput 中五个关键数组

| 数组 | 含义 | GPU 中的消费者 |
| --- | --- | --- |
| `token_ids` | 本微步实际处理的 Token | Embedding |
| `positions` | Token 在各自序列中的绝对位置 | Position Embedding |
| `context_lengths` | 当前 Token 可见的完整上下文长度 | PagedAttention |
| `slot_mapping` | 新 K/V 写入的物理页和页内偏移 | KV Write Kernel |
| `block_tables` | 每个逻辑页对应哪个物理页 | Attention 历史 K/V 读取 |

`slot_mapping` 与 `block_tables` 看似重复，职责不同。前者让写入 Kernel 直接找到一个
新 Token 的目标 Slot；后者让 Attention 遍历整个历史上下文。这个划分与服务框架中
“写入位置”和“可见缓存页”两类元数据一致。

## GPU 内存生命周期

`GPT2CudaModelRunner::Impl` 使用 RAII 管理 CUDA Stream、cuBLAS Handle 和所有设备缓冲。
构造时完成一次权重上传和 KV Cache 分配；每轮执行复用同一组激活、logits 和元数据
Buffer，不在 Token 循环中反复 `cudaMalloc`。

模型级测试配置记录的设备内存为：

```text
GPT-2 权重：497,903,616 bytes
3 个 KV Block：3,538,944 bytes
Batch 2 激活与 logits：494,592 bytes
整次动态请求测试上传的调度元数据：1,312 bytes
```

正式服务 Benchmark 配置 16 个 KV Block 和 Batch 4，对应 KV Cache 18,874,368 bytes、
激活与 logits 989,184 bytes。权重和中间激活不回传 CPU；正常执行只回传每个有效请求
的 Greedy Token ID。`last_logits_for_testing()` 是正确性测试专用调试接口。

## 一步 Transformer 怎样执行

每个单 Token 微批次依次执行：

1. GPU Embedding Kernel 读取 Token Embedding 和绝对 Position Embedding。
2. LayerNorm Kernel 在一个 Block 内完成均值和方差归约。
3. cuBLAS 计算 QKV Projection，Bias Kernel 加偏置。
4. Split Kernel 将 `[B, 3C]` 拆为连续的 Q、K、V。
5. CUDA PagedAttention 根据 Slot Mapping 写新 K/V，根据 Block Table 读取历史。
6. cuBLAS 执行 Attention Projection，随后做 Residual Add。
7. 执行第二个 LayerNorm、FC、GELU、Projection 和 Residual Add。
8. 重复 12 层，再执行 Final LayerNorm 和共享词嵌入矩阵的 LM Head。
9. GPU Argmax 对 50,257 个有效词表 logits 归约，只复制 Token ID 到 CPU。

权重在 checkpoint 中按行主序 `[output, input]` 保存。cuBLAS 使用列主序接口，因此调用
`cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, ...)`，把同一段内存解释成转置后的
`[output, input] × [input, batch]`，输出内存在 C++ 侧仍可视为 `[batch, output]`。

## 正确性测试

`dev/cuda/test_gpt2_cuda_model_runner.cu` 使用真实 GPT-2 124M 权重，覆盖：

- Prompt 长度 17 的 16→17 跨页；
- Batch 大小变化和 Prefill/Decode 混合；
- 请求结束后释放 Block，第三个请求强制复用旧物理页；
- GPU 设备 Argmax 与 CPU 完整前缀 Greedy Argmax；
- 最后一个微批次的全部 50,257 个有效 logits；
- 每轮 H2D 字节数只包含五类调度元数据。

结果：

```text
max_abs_logit_error=0.00025177
all greedy tokens equal CPU full-prefix reference
memcheck: 0 errors
racecheck: 0 hazards
```

运行：

```bash
make GPU_COMPUTE_CAPABILITY=86 test_gpt2_cuda_model_runner
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  ./test_gpt2_cuda_model_runner
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  compute-sanitizer --tool memcheck ./test_gpt2_cuda_model_runner
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  compute-sanitizer --tool racecheck ./test_gpt2_cuda_model_runner
```

## 端到端服务 Benchmark

工作负载与任务 02 的 CPU Benchmark 相同：GPT-2 124M，4 个请求在计时起点同时到达，
Prompt 长度 8/16/24/32，每请求生成 4 Token。Engine 初始化和权重 H2D 不计时；先执行
一次 Warmup，再正式重复 3 次。Benchmark 会独立执行 CPU 完整前缀 Greedy Reference，
任一生成 Token 不一致就直接失败。

| 路径 | 中位总时间 | 输出吞吐 | TTFT P50/P95 | TPOT P50/P95 | 请求延迟 P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU Continuous | 2074.0 ms | 7.714 tok/s | 1200.8 / 1812.5 ms | 276.3 / 276.3 ms | 2029.8 / 2067.5 ms |
| CUDA Continuous | 54.107 ms | 295.708 tok/s | 30.912 / 47.232 ms | 1.397 / 19.200 ms | 52.900 / 53.926 ms |

固定负载下 CUDA 吞吐约为 CPU 的 38.3 倍。这个数字只比较本项目同一 GPT-2 权重、
Token、调度配置和输出长度，不用于声称相对 vLLM 或其他推理引擎的性能。

原始结果：

- `benchmark/results/gpt2_cuda_rtx3090.json`
- `benchmark/results/gpt2_cuda_rtx3090.csv`

复现：

```bash
make GPU_COMPUTE_CAPABILITY=86 benchmark_gpt2_cuda_serving
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 \
  ./benchmark_gpt2_cuda_serving --repeats 3 \
  --json benchmark/results/gpt2_cuda_rtx3090.json \
  --csv benchmark/results/gpt2_cuda_rtx3090.csv
```

## Nsight Systems 分析

Profile 包含一次 Warmup 和一次正式运行，共执行 86 个单 Token 微批次。GPU Kernel
时间分布中，cuBLAS GEMV/GEMM 类 Kernel 合计约 63%，PagedAttention 为 8.7%，
LayerNorm 为 8.1%，Bias Kernel 为 6.3%，Argmax 为 5.4%。两次运行共启动 17,576 个
Kernel，平均每个微批次约 204 次 Launch。

这组数据给出两个直接结论：

1. 当前 Prompt 被拆成 T=1 微步，大量线性层退化为小 Batch GEMV，Prefill 没有利用大矩阵乘。
2. 小算子和 Launch 数量过多，Bias/Residual/LayerNorm Fusion 与 CUDA Graph 有实际依据。

Nsight 原始汇总位于：

- `benchmark/results/gpt2_cuda_nsys_cuda_gpu_kern_sum.csv`
- `benchmark/results/gpt2_cuda_nsys_cuda_api_sum.csv`

复现命令：

```bash
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,cublas --sample=none --cpuctxsw=none \
  --output=/tmp/gpt2_cuda_profile \
  ./benchmark_gpt2_cuda_serving --repeats 1 \
  --json /tmp/gpt2_cuda_profile.json --csv /tmp/gpt2_cuda_profile.csv

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  /tmp/gpt2_cuda_profile.nsys-rep
```

Profiler 会增加运行时间，因此不能把 Profile 中的吞吐当成正式性能数字。

## 当前限制和下一步

当前版本完成了端到端系统接入，但仍是正确性优先的 FP32 基线：

- Prefill 仍拆为 T=1 微步；
- 权重和 KV Cache 只有 FP32；
- Bias、Residual、LayerNorm、GELU 等为独立 Kernel；
- 每轮把完整 Block Table 从 CPU 更新到 GPU；
- 没有 Prefix Cache、抢占、CUDA Graph 和 Tensor Parallel。

下一任务优先实现多 Token Prefill。原因是它既能把线性层从 GEMV 提升到 GEMM，也能显著
减少 Prompt 阶段的 Kernel Launch。完成正确性和 Benchmark 后，再依次做低精度、融合、
CUDA Graph、Prefix Cache 和抢占。

该后续任务已经完成，结果见 `task_05_multi_token_prefill_zh.md`。

面试时应能解释：为什么 Slot Mapping 和 Block Table 都需要；为什么权重上传不应计入
每请求 Decode 时间；为什么只有 Token ID 可以回 CPU；以及 Nsight 数据为什么指向
Multi-Token Prefill 和 Fusion，而不是继续只优化已经占 8.7% 的 PagedAttention。
