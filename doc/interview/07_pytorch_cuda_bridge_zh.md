# 第 7 篇：从 PyTorch Tensor 操作逐段读懂 C++ / CUDA

[返回学习目录](README.md) · 前置：[完整请求执行过程](06_request_walkthrough_zh.md) · 下一篇：[从测试反推正确性](08_tests_as_spec_zh.md)

这篇从你熟悉的 `F.linear`、Tensor 索引、Softmax 和 `argmax` 出发，解释项目代码为何使用页表、扁平地址、持久缓冲和显式 kernel 调用。

每个操作按“数学表达 → 可运行 PyTorch → CUDA 地址或调用点 → 正确性检查 → 面试追问”展开。先学习数学与布局，再学习线程分工。项目使用 GPT-2 的学习范围，不把它解释成已经实现 RoPE、GQA 或 FlashAttention。

## 1. 可运行材料与验证范围

程序：[pytorch_operator_bridge.py](examples/pytorch_operator_bridge.py)。可单独运行四个实验：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda activate zyf1
python doc/interview/examples/pytorch_operator_bridge.py --case qkv
python doc/interview/examples/pytorch_operator_bridge.py --case kv
python doc/interview/examples/pytorch_operator_bridge.py --case attention
python doc/interview/examples/pytorch_operator_bridge.py --case head
# 也可以一次运行：
python doc/interview/examples/pytorch_operator_bridge.py --case all
```

全部使用 CPU、float64 和固定种子；不下载权重。程序对照两种数学或索引表达，已保存 [运行输出](examples/pytorch_operator_bridge_output.txt)。它没有把 Python 的逐层结果逐元素导出与 CUDA 对比；真实 CUDA 的数值证据需看项目原有测试。

本篇 Python 小实验处理一个 Attention 层和独立投影算子，不是一个完整 GPT-2 forward。上一篇的 CUDA 小模型则执行完整两层网络。分清这两种实验分别回答什么问题。

## 2. 固定一组形状，避免边看边换含义

沿用上一篇第 3 轮的逻辑输入：A 已计算 17 行、本轮 1 行；B 已计算 15 行、本轮 4 行。为了暴露错误寻址，本篇故意改成打乱的物理页表，并用更小的张量便于打印。

| 记号 | 本篇取值 | 含义 |
| --- | --- | --- |
| B | 2 | 请求数量 |
| N | 5 | 本轮输入 Token 行数 |
| R | 2 | 最终需要采样的行数 |
| C | 8 | hidden channels |
| H / D | 2 / 4 | 头数 / 每头维度，C=H×D |
| L | 2 | KV 池层数；实验写入 layer=1 |
| P | 16 | 每页 Token 数 |
| V / Vp | 13 / 16 | 有效词表 / 分配时补齐的词表 |
| A 页表 | `[5,2]` | 对应逻辑页 0、1 |
| B 页表 | `[3,7]` | 对应逻辑页 0、1 |

**本篇的物理页号与上一篇不同。** 逻辑 position、context 和 sample rows 相同；slot 需要按新页表重算。两份程序也不是同一组模型权重。

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 16—25 行](examples/pytorch_operator_bridge.py#L16)。

```python
    # A 已计算17个、本轮1个；B 已计算15个、本轮4个。
    owners = torch.tensor([0, 1, 1, 1, 1])
    positions = torch.tensor([17, 15, 16, 17, 18])
    tables = torch.tensor([[5, 2], [3, 7]])
    row_tables = tables[owners]
    slots = row_tables.gather(1, (positions // PAGE)[:, None]).squeeze(1) * PAGE + positions % PAGE
    x, w, bias = rand(5, CHANNELS), rand(3 * CHANNELS, CHANNELS), rand(3 * CHANNELS)
    normalized = F.layer_norm(x, (CHANNELS,), eps=1e-5)
    projected = F.linear(normalized, w, bias)
    q, k, v = [part.reshape(5, HEADS, DIM).contiguous() for part in projected.chunk(3, -1)]
```

`owners=[0,1,1,1,1]` 是 Python 参考中的请求下标，不是项目的 request ID。项目 request ID 可以是 1、2 或任意允许的整数，而 Tensor 下标必须处在对应维度范围内。

## 3. 一张形状表贯穿本篇

| 操作 | 输入 | 输出 | 与项目对应 |
| --- | --- | --- | --- |
| Embedding | IDs/positions `[N]` | `[N,C]` | `embedding_kernel` |
| LayerNorm | `[N,C]` | `[N,C]` | `layernorm_kernel` |
| QKV Linear | `[N,C]` 与 `[3C,C]` | `[N,3C]` | `matmul` |
| QKV Split | `[N,3C]` | 三份 `[N,H,D]` | `split_qkv_kernel` |
| KV 写入 | K/V `[N,H,D]` | 修改缓存 `[pages,L,H,P,D]` | `write_kv_cache_kernel` |
| Attention | Q 与对应逻辑历史 K/V | `[N,H,D]` | `paged_attention_kernel` |
| 输出投影 / MLP | `[N,C]` | `[N,C]` | `matmul` 与激活、残差 |
| Gather | hidden `[N,C]` 与 rows `[R]` | `[R,C]` | `gather_sample_rows_kernel` |
| LM Head | `[R,C]` 与 `[Vp,C]` | `[R,Vp]` | `logits_matmul` |
| Greedy | 有效列 `[R,V]` | `[R]` | `argmax_kernel` |

这里的 `[N,H,D]` 与常见 `[B,H,T,D]` 是不同的组织方式：N 已经把多请求本轮的 Token 展开，每行通过元数据找回自己的历史上下文。

## 4. Embedding：Token ID 和 position 分别查哪张表

你熟悉的表达是：

```python
# 概念表达；wte=[Vp,C]，wpe=[max_context,C]
x = wte[token_ids] + wpe[positions]  # [N,C]
```

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 322—335 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L322)。

```cpp
template <typename T>
__global__ void embedding_kernel(
    T* output, const int* token_ids, const int* positions,
    const T* token_embeddings, const T* position_embeddings,
    int batch_size, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch_size * channels;
    if (index >= count) return;
    const int row = index / channels;
    const int channel = index % channels;
    output[index] = from_float<T>(
        to_float(token_embeddings[token_ids[row] * channels + channel]) +
        to_float(position_embeddings[positions[row] * channels + channel]));
}
```

`index` 是输出扁平元素编号。`row=index/C` 找 Token 行，`channel=index%C` 找通道。输出元素来自 Token embedding 与绝对位置 embedding 的同一通道相加。

调用点：[forward 中的 embedding](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1138)。`positions` 来自 [ModelInput 的 computed+offset](../../mini_vllm/model_input.hpp#L64)，不是当前 Batch 从 0 开始的行号。

第 3 轮第 0 行 A 的位置是 17。如果误填为 0，QKV 投影之前的输入激活就已改变。此时修 Attention 页表不能修复位置编码错误。

**口述句：** Token ID 决定查哪一个词向量，position 决定该 Token 在请求中的绝对位置；Packed 后仍要保留各请求原本的位置语义。

## 5. LayerNorm：为什么归一化不跨请求混合

PyTorch 参考使用：

```python
normalized = F.layer_norm(x, (C,), eps=1e-5)
# 对每行单独计算 mean/variance；实际模型还使用 learned weight 和 bias。
```

项目入口：[layernorm_kernel](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L338)，调用点：[每层 LN1](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1154)、[末尾 LN](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1295)。

先写数学关系：

```text
mean[r] = sum_c x[r,c] / C
variance[r] = sum_c (x[r,c] - mean[r])² / C
y[r,c] = (x[r,c] - mean[r]) / sqrt(variance[r]+epsilon) * gamma[c] + beta[c]
```

归约范围是通道 C，不是 N。因此 A 与 B 的行放在同一个矩阵里，并不会让 LayerNorm 把它们的均值混在一起。Python 小实验省略 learned affine 参数，相当于 gamma=1、beta=0；实际 GPT-2 使用参数缓冲中的对应权重。

本项目多个归约使用 float 累加，但存储可能是 FP16/BF16。Python float64 对照帮助看清公式，不能用它的极小误差当作低精度 CUDA 的承诺。

## 6. QKV Linear：先写矩阵，再看 cuBLAS 参数

```python
# N=5,C=8；weight 按 nn.Linear 的 [out_features,in_features] 组织。
qkv = F.linear(normalized, weight, bias)
# [5,8] @ [8,24] + [24] -> [5,24]
```

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1166—1177 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1166)。

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
```

`matmul` 的三个尺寸按本例解释为 `batch_size=5,input_width=8,output_width=24`。它覆盖全部输入行，不只覆盖 R 个采样行，因为每个输入 Token 都要产生后续层需要的表示和 K/V。

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 990—1008 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L990)。

```cpp
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if constexpr (std::is_same<T, float>::value) {
            check_cublas(cublasSgemm(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, input_width, input, input_width, &beta,
                output, output_width), "cublasSgemm");
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

读这一段时，先忽略 API 名字中的转置标记，只固定目标关系：`Y=XWᵀ`，形状 `[N,O]`。cuBLAS 使用列主序视角，目标内存可看作 `Yᵀ=WXᵀ`，形状 `[O,N]`。

| 同一段内存 | C++ 项目按行主序理解 | cuBLAS 初始列主序视角 |
| --- | --- | --- |
| input | X `[N,C]` | Xᵀ `[C,N]` |
| weight | W `[O,C]` | Wᵀ `[C,O]`，经 OP_T 得 W |
| output | Y `[N,O]` | Yᵀ `[O,N]` |

于是 `m=O,n=N,k=C`，weight/input 的 leading dimension 是 C，output 的 leading dimension 是 O。这里没有为数学转置额外复制一份矩阵；是按 API 的布局规则解释既有内存。

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 82—86 行](examples/pytorch_operator_bridge.py#L82)。

```python
    # 对照 row-major Y=XW^T 与 cuBLAS 列主序视角 Y^T=WX^T。
    row_result = data['normalized'] @ data['w'].T
    column_result = data['w'] @ data['normalized'].T
    torch.testing.assert_close(row_result, column_result.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(row_result + data['bias'], data['projected'], atol=1e-12, rtol=1e-12)
```

Python 这段检查等式两边的数值一致。它解释转置关系，不模拟 cuBLAS 的线程或内部算法。

## 7. Bias 与 QKV Split：一个元素怎样找到来源

Bias 在项目 `matmul` 的 GEMM 之后由独立 kernel 加入，见 [bias 调用分支](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1010)。因此查耗时时可能同时看到 GEMM 和 bias kernel；不能只找一个名字为 Linear 的设备事件。

QKV 拆分的 PyTorch 表达：

```python
q, k, v = qkv.chunk(3, dim=-1)
q, k, v = [part.reshape(N, H, D).contiguous() for part in (q, k, v)]
```

真实 kernel 见 [split_qkv_kernel](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L537)。本次实验逐元素核对三段来源：

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 87—92 行](examples/pytorch_operator_bridge.py#L87)。

```python
    flat = data['projected'].flatten()
    # 用 CUDA split 的扁平地址单独检查 Q、K、V 三段。
    for row in range(5):
        for channel in range(CHANNELS):
            for part, name in enumerate(('q', 'k', 'v')):
                assert flat[row*3*CHANNELS + part*CHANNELS + channel] == data[name].reshape(5, -1)[row, channel]
```

对第 r 行、第 c 通道：Q 从 `r*3C+c` 读，K 从 `r*3C+C+c` 读，V 从 `r*3C+2C+c` 读。之后 `head=c//D`、`dimension=c%D`。

`chunk` 通常提供原张量的视图，参考代码的 `.contiguous()` 明确得到连续存储。项目 CUDA kernel 则把三段写入独立持久缓冲。两者数学结果相同，内存生命周期与分配方式不同。

**追问：能直接把 qkv 指针当成连续 K 吗？** 不能只偏移 C 就假设所有 Token 的 K 紧挨着。原布局每行还有 Q/V，行间步长是 3C；要么像项目一样拆到独立缓冲，要么设计能处理原 stride 的消费者。

## 8. KV 池：为什么是五个维度

本项目每一份 K 或 V 缓存的逻辑形状为：

```text
[physical_pages, layers, heads, tokens_per_page, head_dimension]
```

K 和 V 分别持有一份这样的设备缓冲。逻辑 Token 的完整 K/V 分布在所有层、所有头的对应槽位，不是 slot 指向的单个标量。

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 17—25 行](../../mini_vllm/cuda/paged_attention.cu#L17)。

```cpp
__device__ __forceinline__ std::size_t cache_offset(
    int page, int layer, int head, int page_offset, int dimension,
    int num_layers, int num_heads, int head_size) {
    return ((((
        static_cast<std::size_t>(page) * num_layers + layer) *
            num_heads + head) *
            kPagedAttentionPageSize + page_offset) *
            head_size + dimension);
}
```

这就是行主序多维数组的逐维展开：先选页，再选层，再选头，再选页内 Token，最后选 head dimension。

如果元素类型是 FP16，元素偏移乘 2 才是字节偏移；FP32 乘 4。本篇 Python 使用 float64，因此其底层字节数又不同。**slot 是 Token 槽编号，cache_offset 是单个浮点元素的位置，二者不能互换。**

## 9. 从 position 一步步推到物理元素

本篇第 0 行是 A 的 position=17，页表 `[5,2]`，P=16：

```text
logical_page = 17 // 16 = 1
physical_page = table[1] = 2
page_offset = 17 % 16 = 1
physical_slot = 2*16+1 = 33
```

选定 layer=1、head=1、dimension=3，本篇 L=2,H=2,D=4：

```text
element_offset = ((((2*2+1)*2+1)*16+1)*4+3) = 711
```

`slot=33` 告诉写入 kernel 使用哪一个页内 Token 位置；711 才是选定层、头和维度后在 K 缓冲中的元素索引。

| Packed 行 | 请求 / position | 页表 | slot |
| --- | --- | --- | --- |
| 0 | A / 17 | `[5,2]` | 33 |
| 1 | B / 15 | `[3,7]` | 63 |
| 2 | B / 16 | `[3,7]` | 112 |
| 3 | B / 17 | `[3,7]` | 113 |
| 4 | B / 18 | `[3,7]` | 114 |

从 B 的位置 15 到 16，slot 从 63 跳到 112。只要页表正确，这种不连续是正常结果。

## 10. KV 写入：Python 赋值对应 CUDA 哪几行

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 41—45 行](examples/pytorch_operator_bridge.py#L41)。

```python
def write_pages(data):
    for row, slot in enumerate(data['slots'].tolist()):
        page, offset = divmod(slot, PAGE)
        data['cache_k'][page, LAYER, :, offset, :] = data['k'][row]
        data['cache_v'][page, LAYER, :, offset, :] = data['v'][row]
```

这段循环一次写一个 Token 的所有 head/dimension。在 CUDA 中，用 grid 的 x 维选 Token 行、y 维选 head，再让线程分摊 dimension。

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 64—74 行](../../mini_vllm/cuda/paged_attention.cu#L64)。

```cpp
    const int request = blockIdx.x;
    const int head = blockIdx.y;
    if (request >= batch_size || head >= num_heads) return;

    if (context_lengths[request] <= 0) return;
    const int physical_slot = slot_mapping[request];
    const int physical_block = physical_slot / kPagedAttentionPageSize;
    const int page_offset = physical_slot % kPagedAttentionPageSize;
    const std::size_t source_base =
        (static_cast<std::size_t>(request) * num_heads + head) *
        head_size;
```

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 96—103 行](../../mini_vllm/cuda/paged_attention.cu#L96)。

```cpp
    for (int dimension = threadIdx.x;
         dimension < head_size; dimension += blockDim.x) {
        const std::size_t destination = cache_offset(
            physical_block, layer_index, head, page_offset, dimension,
            num_layers, num_heads, head_size);
        k_cache[destination] = new_k[source_base + dimension];
        v_cache[destination] = new_v[source_base + dimension];
    }
```

`source_base` 来自 `[N,H,D]` 的展开；`destination` 来自 `[pages,L,H,P,D]` 的展开。读写布局不同，因此这里是一种按映射搬运，而不是把整个 K Tensor 连续 memcpy 到缓存末尾。

同一函数中 FP16 且 head size 为偶数时使用 half2 分支，每次处理相邻两个维度。它改变访存和计算组织，不改变逻辑页表公式。先掌握标量分支，再看向量化分支。

实验既检查目标槽写入的新值，也检查目标之外的所有元素保持不变，见 [kv_lesson](examples/pytorch_operator_bridge.py#L97)。这样能够区分“值写对但写到了错误层/额外位置”的问题。

## 11. 为什么还需要一个独立的 dense 参考

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 48—52 行](examples/pytorch_operator_bridge.py#L48)。

```python
def write_dense(data):
    # 独立按请求与逻辑位置写，不复用页表或 slot，避免两条路径同时抄错映射。
    for row, (owner, position) in enumerate(zip(data['owners'].tolist(), data['positions'].tolist())):
        data['dense_k'][owner, position] = data['k'][row]
        data['dense_v'][owner, position] = data['v'][row]
```

dense 参考按 `[请求,逻辑位置,head,dimension]` 存储，不使用页表和物理 slot。这样它不会与待验证分页路径共享同一套寻址计算。

如果两条路径都调用同一个错误的 slot 函数，输出可能相同，但只说明错误被同时复制了。参考实现要尽量在你想验证的维度上独立。

实验 fixture 先装入 A 的前 17 个、B 的前 15 个历史 K/V，再用本轮的新 K/V 更新相应位置。它模拟“历史已存在，本轮追加”的算子前提，不宣称这些历史随机 K/V 来自一个完整模型的前几轮 forward。

## 12. Dense Attention：把因果 mask 写出来

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 55—61 行](examples/pytorch_operator_bridge.py#L55)。

```python
def dense_attention(data):
    keys = data['dense_k'][data['owners']].permute(0, 2, 1, 3)  # [N,H,T,D]
    values = data['dense_v'][data['owners']].permute(0, 2, 1, 3)
    scores = torch.einsum('nhd,nhtd->nht', data['q'], keys) / math.sqrt(DIM)
    visible = torch.arange(keys.shape[2])[None, :] <= data['positions'][:, None]
    scores = scores.masked_fill(~visible[:, None, :], float('-inf'))
    return torch.einsum('nht,nhtd->nhd', scores.softmax(-1), values)
```

逐行读形状：

1. 按 owners 找到各行所属请求的连续 KV，得到 `[N,H,T,D]`。
2. Q 是 `[N,H,D]`，与各自历史 K 点积得到 `[N,H,T]`。
3. `visible[row,t]` 仅在 `t<=positions[row]` 时成立。
4. 不可见位置的分数设为负无穷，Softmax 后权重为零。
5. 权重与 V 相乘归约 T，得到 `[N,H,D]`。

B 的四个 query 位置是 `[15,16,17,18]`，而 key 位置从 0 开始。不能对一个 `[4,19]` 分数矩阵直接使用左上角 `tril`；那会把第一行错误限制为只看位置 0。mask 必须使用绝对 query position。

这里给每个 query 复制逻辑上的请求 KV 视图，是为了公式清晰；它不是建议实际服务系统物化这么大的重复张量。

## 13. Paged Attention：用读取范围实现同一约束

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 64—77 行](examples/pytorch_operator_bridge.py#L64)。

```python
def paged_attention(data):
    output = []
    for row, position in enumerate(data['positions'].tolist()):
        logical = torch.arange(position + 1)
        pages = data['row_tables'][row, logical // PAGE]
        offsets = logical % PAGE
        # 高级索引输出 [context,H,D]，再变成 [H,context,D]。
        keys = data['cache_k'][pages, LAYER, :, offsets, :].transpose(0, 1)
        values = data['cache_v'][pages, LAYER, :, offsets, :].transpose(0, 1)
        scores = (data['q'][row, :, None, :] * keys).sum(-1) / math.sqrt(DIM)
        weights = (scores - scores.max(-1, keepdim=True).values).exp()
        weights = weights / weights.sum(-1, keepdim=True)
        output.append((weights[:, :, None] * values).sum(1))
    return torch.stack(output)
```

分页参考对每行构造 `logical=0…position`，用页表把这些位置还原成物理元素。因此它从一开始就没有读取未来 key，不必额外建立含未来位置的分数矩阵再 mask。

真实 CUDA 读取处：

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 125—138 行](../../mini_vllm/cuda/paged_attention.cu#L125)。

```cpp
    const std::size_t query_base =
        (static_cast<std::size_t>(request) * num_heads + head) *
        head_size;
    const float scale = rsqrtf(static_cast<float>(head_size));

    float local_max = -FLT_MAX;
    for (int token = thread; token < context_length;
         token += blockDim.x) {
        const int physical_block =
            block_tables[
                request * max_blocks_per_sequence +
                token / kPagedAttentionPageSize];
        const int page_offset = token % kPagedAttentionPageSize;
        float dot = 0.0f;
```

`context_length` 来自当前行的 `position+1`。每个线程按 `thread,thread+blockDim.x,...` 分摊历史 Token，当前标量路径再在 head dimension 上求点积。

这里 `request=blockIdx.x` 在 Packed 路径中是 Token 行。B 的四个 query 分别有自己的页表行和 context，但它们的页表内容属于同一个请求。

**可背的等价关系：** Dense 用 mask 把未来位置权重变成零；本项目分页 kernel 通过每行 context 限制读取区间，实现相同的因果可见范围。

## 14. Softmax 与 V 归约：别只记一个公式名字

在得到每个历史位置的点积分数后，项目先找最大值，再计算指数及总和。

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 185—206 行](../../mini_vllm/cuda/paged_attention.cu#L185)。

```cpp
    const float maximum = reduction[0];
    // Every thread must finish reading the maximum before reduction[] is
    // reused for the softmax sum below.
    __syncthreads();

    float local_sum = 0.0f;
    for (int token = thread; token < context_length;
         token += blockDim.x) {
        const float weight = expf(scores[token] - maximum);
        scores[token] = weight;
        local_sum += weight;
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / reduction[0];
    __syncthreads();
```

减去同一行最大值不改变归一化后的数学结果，但使指数的输入更适合数值计算。`reduction[]` 在不同阶段复用，因此同步既保护归约，也保护“上一个值已被所有线程读完再覆盖”的时序。

接着按 dimension 累加各位置 V：

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 238—253 行](../../mini_vllm/cuda/paged_attention.cu#L238)。

```cpp
    for (int dimension = thread; dimension < head_size;
         dimension += blockDim.x) {
        float value_sum = 0.0f;
        for (int token = 0; token < context_length; ++token) {
            const int physical_block = block_tables[
                request * max_blocks_per_sequence +
                token / kPagedAttentionPageSize];
            const int page_offset = token % kPagedAttentionPageSize;
            const std::size_t offset = cache_offset(
                physical_block, layer_index, head, page_offset,
                dimension, num_layers, num_heads, head_size);
            value_sum += scores[token] * inverse_sum *
                to_float(v_cache[offset]);
        }
        out[query_base + dimension] = from_float<T>(value_sum);
    }
```

数学上是 `output[h,d]=sum_t probability[h,t]*V[h,t,d]`。代码中的 `scores` 在 Softmax 阶段已被改成未归一化的指数权重，乘 `inverse_sum` 后才是概率；不要把同名缓冲始终理解成原始点积分数。

本 kernel 在共享内存保存一个 query/head 的分数数组，没有实现 FlashAttention 的分块在线 Softmax。理解到这一层后，才能准确说明当前实现与进一步优化的差别。

## 15. 先写全部新 KV，为什么不会看到未来

源码/记录：[mini_vllm/cuda/paged_attention.cu，第 278—291 行](../../mini_vllm/cuda/paged_attention.cu#L278)。

```cpp
    const dim3 grid(batch_size, num_heads);
    write_kv_cache_kernel<T><<<grid, kThreads, 0, stream>>>(
        new_k, new_v, k_cache, v_cache, context_lengths, slot_mapping,
        batch_size, num_layers, layer_index, num_heads, head_size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    const std::size_t shared_memory =
        static_cast<std::size_t>(max_context_length) * sizeof(float);
    paged_attention_kernel<T><<<grid, kThreads, shared_memory, stream>>>(
        q, k_cache, v_cache, block_tables, context_lengths, out,
        batch_size, num_layers, layer_index, num_heads, head_size,
        max_blocks_per_sequence, max_context_length);
    return cudaGetLastError();
```

两个 kernel 在同一 stream 上按顺序执行：先完成新 KV 写入，再运行 Attention 读取。本轮可以先把 B 的位置 15—18 都写进池中，但 B 位置 15 的读取循环仍只到 `context=16`。

这里有两条互补约束：执行顺序确保“需要读的当前 KV 已写完”；每行 context 确保“已经写入的未来 KV 不会被读”。只有前者会发生未来泄漏，只有后者则可能读取尚未写好的当前值。

实验专门修改 B 的未来 V：

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 124—136 行](examples/pytorch_operator_bridge.py#L124)。

```python
    # 修改 B 位置16..18的未来V：不应影响 B 位置15，也不应影响 A。
    for position in (16, 17, 18):
        data['dense_v'][1, position] += 10000
        page = data['tables'][1, position // PAGE].item()
        data['cache_v'][page, LAYER, :, position % PAGE, :] += 10000
    changed = paged_attention(data)
    torch.testing.assert_close(changed[:2], paged[:2], atol=0, rtol=0)
    assert (changed[2:] - paged[2:]).abs().max() > 1
    torch.testing.assert_close(changed, dense_attention(data), atol=1e-10, rtol=1e-12)
    # 教学反例：把正确页表替换为“物理页从0连续排布”，必须被全量输出比较发现。
    data['row_tables'] = torch.tensor([[0, 1]]).repeat(5, 1)
    wrong = paged_attention(data)
    assert not torch.allclose(wrong, changed)
```

观察 A 行与 B 最早一行不变，B 后续行发生变化。这同时检查请求隔离和同请求因果范围。错误页表的反例也必须产生不同结果，证明 fixture 对这种错误有区分能力。

注意这里扰动的是独立 Attention 输入中的 V，不是直接声称修改完整模型中某个未来 Token 后所有中间 K/V 都按同一方式变化；完整模型的因果实验见 [入门生成实验](../from_pytorch/01_generation_and_kv.md)。

## 16. Attention 后怎样接回普通 Transformer

返回 `[N,H,D]` 可以按连续通道视为 `[N,C]`。随后是 Attention 输出投影、残差、第二个 LayerNorm、MLP、第二个残差。

```python
# 非融合路径的数学草图，不是本篇小实验的完整实现。
x = x + F.linear(attended.reshape(N, C), w_out, b_out)
z = F.layer_norm(x, (C,), ln2_weight, ln2_bias, eps=1e-5)
z = F.linear(z, w_fc, b_fc)              # [N,4C]
z = F.gelu(z, approximate='tanh')
x = x + F.linear(z, w_proj, b_proj)     # [N,C]
```

项目调用点：[Attention 输出投影](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1191)、[MLP 第一层](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1235)、[GELU](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1243)、[MLP 第二层](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1255)。

PyTorch 草图用新的 Tensor 变量表达数学关系，项目复用 `residual_a_`、`residual_b_`、`normalized_` 等工作缓冲。读覆盖写入时，要问旧内容的最后一个消费者是否已经执行。

开启融合时，部分 residual 与下一处 LayerNorm 在一个 kernel 中产生两个输出。数学流程和低精度舍入都需要结合具体分支验证，不能仅凭 kernel 数少就认定性能或精度更好。

## 17. Gather：为何选择的是 hidden 行

当主体模型与 final LayerNorm 完成后，得到 `[N,C]`。本轮只有 A 的最后输入行和 B 的最后输入行需要预测下一个 Token，rows=`[0,4]`。

源码/记录：[doc/interview/examples/pytorch_operator_bridge.py，第 144—149 行](examples/pytorch_operator_bridge.py#L144)。

```python
    hidden = F.layer_norm(data['x'] + paged_attention(data).reshape(5, CHANNELS), (CHANNELS,), eps=1e-5)
    rows = torch.tensor([0, 4], dtype=torch.long)
    full = F.linear(hidden, data['wte'])
    gathered = hidden.index_select(0, rows)
    pruned = F.linear(gathered, data['wte'])
    torch.testing.assert_close(pruned, full.index_select(0, rows), atol=1e-12, rtol=1e-12)
```

原始计算是 `H[N,C] @ Wᵀ[C,Vp]`；裁剪后先选择 H 的 R 行，再计算 `[R,C] @ [C,Vp]`。

因为线性层的各行彼此独立，在相同数学输入下，有：

```text
linear(H)[rows] = linear(H[rows])
```

这条等价关系只说明末尾投影可裁剪；不能据此把前面用于建立 KV 的输入行一并删除。

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 638—647 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L638)。

```cpp
// Gather 只选中每个已完成输入请求的最后一行，避免 Prompt 全行 LM Head。
template <typename T>
__global__ void gather_sample_rows_kernel(
    T* output, const T* input, const int* rows, int count, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * channels) return;
    const int row = index / channels;
    const int channel = index % channels;
    output[index] = input[static_cast<std::size_t>(rows[row]) * channels + channel];
}
```

对输出元素 index，先算压缩行 `row=index/C`、通道 `channel=index%C`，再通过 `rows[row]` 找原始 hidden 行。row=1 并不意味着取原矩阵第 1 行；本例实际取原第 4 行。

## 18. LM Head 的权重、输出精度与有效词表

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1302—1317 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1302)。

```cpp
        if (num_logit_rows > 0) {
            const T* lm_input = normalized_.get<T>();
            if (config_.enable_sample_row_pruning) {
                gather_sample_rows_kernel<T><<<
                    blocks_for(num_logit_rows * channels), kThreads, 0, stream_.get()>>>(
                    sampled_hidden_.get<T>(), normalized_.get<T>(), sample_rows_.get(),
                    num_logit_rows, channels);
                check_last_kernel("gather_sample_rows_kernel");
                lm_input = sampled_hidden_.get<T>();
            }
            logits_matmul(lm_input, parameters_view.wte,
                          num_logit_rows, channels, config_.padded_vocab_size);
            argmax_kernel<<<num_logit_rows, kThreads, 0, stream_.get()>>>(
                logits_.get(), sampled_token_ids_.get(), num_logit_rows,
                config_.vocab_size, config_.padded_vocab_size);
            check_last_kernel("argmax_kernel");
```

这里使用 `parameters_view.wte`，即与 Token embedding 共享的权重。它在内存中是 `[Vp,C]`；传给 `logits_matmul` 的输出宽度是 padded vocab size。

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 1039—1048 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1039)。

```cpp
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

即使输入和权重是 FP16/BF16，logits 输出缓冲仍是 FP32。Python 小实验全部 float64，因此它验证的是行选择与线性关系；低精度误差阈值需要独立考虑。

输出矩阵有 Vp 列不代表每列都是有效 Token。Argmax 的扫描范围是 V：

源码/记录：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 607—617 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L607)。

```cpp
    float best_value = -FLT_MAX;
    int best_index = 0;
    const float* row_logits = logits +
        static_cast<std::size_t>(row) * padded_vocab_size;
    for (int index = thread; index < vocab_size; index += blockDim.x) {
        const float value = row_logits[index];
        if (value > best_value ||
            (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
```

本例 V=13、Vp=16。实验把 padding 列人为设得很大，验证只在有效列中取最大值仍不受影响；如果把扫描范围误改为 Vp，就可能返回非法 Token。

## 19. R=0 时，要区分数学空张量和实际执行分支

Python 可以写：

```python
empty_logits = F.linear(hidden[:0], weight)  # [0,Vp]
```

这解释零行输出的形状。项目实现则在 `if (num_logit_rows > 0)` 下跳过 Gather、LM Head 和 Argmax 的提交，不依赖 cuBLAS 对零规模调用的行为。

但主体网络在这个判断之前已运行，KV 已被计算。Runner 最后仍同步 stream，返回按请求排列的 `-1`。因此“零采样行”不是“零输入行”，也不是整个 step 没工作。

## 20. 为什么不能每轮随意新建设备缓冲

你在 PyTorch 中写 `index_select` 通常不用显式管理内存地址；框架负责张量对象和分配。项目 Runner 在构造时分配足够大的设备缓冲，每轮只更新有效前缀及元数据。

入口：[持久缓冲构造](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L689)、[元数据复制](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1355)、[Graph 查找](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1106)。

开启 CUDA Graph 后，这些地址还会成为捕获执行计划的一部分。保持地址有效与更新地址中的内容是两件事：相同 `(N,R)` 可以使用已有图，但 Token IDs、positions、页号、sample rows 仍要每轮刷新。

Python 参考的 `.contiguous()`、高级索引和新 Tensor 是为了让语义易读。不能把它们的分配行为直接当作项目的高性能实现建议。

## 21. 从代码理解形成四段面试短答

**QKV 与布局：** 我先按 PyTorch 的 `Y=XWᵀ+b` 写出形状，再解释 cuBLAS 的列主序视角。QKV 是每个输入 Token 的三组投影，项目拆到三个连续缓冲，再按 `[N,H,D]` 交给 Attention。

**分页寻址：** 逻辑位置先经页表得到物理页，加页内偏移得到 Token slot；再结合 layer/head/dimension 得到 K 或 V 缓冲中的元素地址。slot 不是字节地址，也不是完整 K/V 向量。

**因果与隔离：** Packed 后每行仍保留所属请求的页表和逻辑 context。同请求的未来位置受 context 限制，其他请求的 KV 由不同页表隔离。先写本轮 KV 再读的 stream 顺序保证当前值有效。

**采样行优化：** 主体模型仍处理 N 行，只有完成当前输入的请求末行需要 LM Head。先 Gather R 行再投影，结果对应完整投影的相同行；需要同步更新行索引、输出映射和 Graph 形状键。

## 22. 闭卷推导与参考答案

**题 1：B 的 position=16、页表 `[3,7]`，为什么不是 slot=64？**

position=16 位于逻辑页 1，物理页由 table[1]=7 决定，slot=7×16+0=112。64 是错误假设“下一物理页紧接页 3”后得到的值。

**题 2：N=5、C=8、R=2、Vp=16，Gather 与 LM Head 的输出各多大？**

Gather 为 `[2,8]`，16 个元素；LM Head 为 `[2,16]`，32 个元素。完整投影输出为 `[5,16]`，80 个元素。这里只数对应张量，不等于整个模型显存减少比例。

**题 3：QKV 的 K 部分能直接用 `qkv[C:]` 表达吗？**

如果对扁平数组切片，这会混入后续 V 和下一行 Q。正确选择每行中间 C 个通道，需要保留 3C 的原行 stride，或显式复制到连续 K 缓冲。

**题 4：为什么模型中要对每一层分别缓存 KV？**

每层 K/V 由该层输入表示与投影参数产生。一个逻辑 Token 在不同层的 K/V 不相同。本项目布局保留 layer 维，`layer_index` 决定当前写入和读取哪层。

**题 5：两个输出的 Argmax 相同，能说明 LM Head 没算错吗？**

不能。例如所有 logits 同加 100，Argmax 保持不变。实验实际构造了这种情况。验证数值结果应比较有效词表的 logits，并用合理误差口径判断。

**题 6：本篇 Python Attention 误差约 1e−15，能用于简历声称 CUDA 达到这个误差吗？**

不能。它是 CPU float64 两种语义表达之间的差异。真实 CUDA 模型有不同存储、归约、GEMM 路径及累积误差，需使用对应测试实测记录。

## 23. 三次学习的检查清单

| 次序 | 必须会讲 | 必须会算 | 选读 |
| --- | --- | --- | --- |
| 第一轮 | Embedding、QKV、Split 数据来源 | 三个矩阵形状、一个 K 元素来源 | cuBLAS leading dimension |
| 第二轮 | slot 与完整元素地址、因果读取 | B 跨页的 slot、偏移 711 | half2 与 shared memory 归约 |
| 第三轮 | Gather、LM Head、有效词表 | N/R/V/Vp 的形状与映射 | CUDA Graph 地址生命周期 |

下一篇会从这些关系反过来读测试：如果把一个公式写错，哪个断言应先失败？如果没有断言失败，说明哪些验证还不够？
