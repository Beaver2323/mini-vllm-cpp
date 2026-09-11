# 开发任务 03：CUDA PagedAttention Decode

本篇把 PyTorch 的 `softmax(QKᵀ / sqrt(D)) @ V` 对照到真实 CUDA 源码。第一遍先读 FP32 标量路径；代码中的 FP16 分支来自任务 06，之后再学。

学习目标：给定一个逻辑 Token 位置，算出 K/V 数组偏移；解释每一阶段由哪些 CUDA 线程处理，以及为什么需要同步。

## 1. 先确定这个算子接收什么

入口文件是 [paged_attention.cuh](../mini_vllm/cuda/paged_attention.cuh)，实现是 [paged_attention.cu](../mini_vllm/cuda/paged_attention.cu)，独立正确性测试是 [test_paged_attention.cu](../dev/cuda/test_paged_attention.cu)。

不要先读整个 GPT-2。这个独立测试已经构造了 Q、新 K/V、历史缓存和页表，足够理解分页 Attention。

| 参数 | 逻辑形状 | 用途 |
| --- | --- | --- |
| `q/new_k/new_v` | `[N,H,D]` | 本次新增的 N 个输入行 |
| `k_cache/v_cache` | 各为 `[pages,L,H,16,D]` | 所有请求跨层保留的 KV |
| `slot_mapping` | `[N]` | 每个新 K/V 的写入槽 |
| `block_tables` | `[N,max_blocks]` | 每行历史 Token 的读地址 |
| `context_lengths` | `[N]` | 每个 Query 的可见长度 |
| `out` | `[N,H,D]` | Attention 输出 |

源代码变量名 `request` 表示 `blockIdx.x` 对应的**输入行**。最初 Decode 一请求一行，看起来等于请求号；packed Prefill 中同一请求有多行，不能继续按请求数解释它。

## 2. 调用顺序：先写新 KV，再读历史与当前 KV

```text
GPT2CudaModelRunner 前向的一层
  ├─ QKV projection / split
  └─ paged_attention_decode → paged_attention_decode_impl（同一 stream）
      ├─ write_kv_cache_kernel：写当前输入的 K/V
      └─ paged_attention_kernel
          ├─ 通过 block table 读 K，计算 scores
          ├─ max reduction
          ├─ exp / sum reduction
          └─ 通过 block table 读 V，加权求和
```

源码：[mini_vllm/cuda/paged_attention.cu，第 278—291 行](../mini_vllm/cuda/paged_attention.cu#L278)。以下为当前文件的原样摘录。

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

上面 launch 的 `grid=(N,H)`，每块处理一行的一个头；第二个 kernel 的动态共享内存按最大上下文分配。

同一 stream 上两个 kernel 有顺序保证，第二个可以看到第一个的写入。kernel 内的 `__syncthreads()` 只同步同一个线程块，不能用来替代两个独立 kernel 之间的调度关系。

packed 情况下，本轮所有新 K/V 都先写入，但每个 Query 只读取自己的 `context_length` 范围，因此未来 Token 即使已写入也不会被看见。任务 05 会把这个因果关系展开。

## 3. 五维布局如何压成一个指针偏移

源码：[mini_vllm/cuda/paged_attention.cu，第 17—25 行](../mini_vllm/cuda/paged_attention.cu#L17)。以下为当前文件的原样摘录。

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

这是 C/C++ 行优先连续布局，最右边 `dimension` 变化最快。可以逐层理解为：

```text
先跨 page：         page * L
再选 layer：       page * L + layer
再跨 head：       (page * L + layer) * H + head
再选页内 token：  上式 * 16 + page_offset
再选 head 维度：  上式 * D + dimension
```

教学手算：`L=2,H=2,D=4`，要读物理页 3、层 1、头 0、页内位置 2、维度 1：

```text
offset = ((((3*2+1)*2+0)*16+2)*4+1)
       = 905 个元素
FP32 字节偏移 = 905*4 = 3620
FP16 字节偏移 = 905*2 = 1810
```

`cache_offset` 返回元素偏移。调用方若再把它当字节偏移，或在指针算术前多乘一次 `sizeof(T)`，会产生错误地址。

K 与 V 是两份独立数组，公式里没有“选择 K/V”的额外维度。任务 11 的传输代码因此分别复制 K 页和 V 页。

## 4. slot_mapping 只管当前写入

源码：[mini_vllm/cuda/paged_attention.cu，第 64—74 行](../mini_vllm/cuda/paged_attention.cu#L64)。以下为当前文件的原样摘录。

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

源码：[mini_vllm/cuda/paged_attention.cu，第 96—103 行](../mini_vllm/cuda/paged_attention.cu#L96)。以下为当前文件的原样摘录。

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

`physical_slot / 16` 找到物理页，`%16` 找到页内位置。随后把本行本头的 D 个维度写过去。

例如逻辑页表 `[5,2]`，当前位置 17：

| 值 | 结果 |
| --- | ---: |
| 逻辑页 `17/16` | 1 |
| 物理页 `table[1]` | 2 |
| 页内偏移 `17%16` | 1 |
| slot | 33 |

kernel 不需要知道请求字符串或 Prompt 文本；它只依赖调度层已经构造正确的映射。出现跨请求污染时，既要检查 CUDA 偏移，也要检查上游是否把同一个可写槽分给了两个输入行。

标量写入中线程 `t` 处理维度 `t, t+128, ...`。GPT-2 的 `D=64` 时只需前 64 个线程写数值；其余线程并不代表还有额外 64 个头。

## 5. QK：一个线程负责一个历史 Token 的完整点积

源码：[mini_vllm/cuda/paged_attention.cu，第 118—138 行](../mini_vllm/cuda/paged_attention.cu#L118)。以下为当前文件的原样摘录。

```cpp
    extern __shared__ float scores[];
    __shared__ float reduction[kThreads];

    const int context_length = context_lengths[request];
    if (context_length <= 0 || context_length > max_context_length) {
        return;
    }
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

源码：[mini_vllm/cuda/paged_attention.cu，第 162—174 行](../mini_vllm/cuda/paged_attention.cu#L162)。以下为当前文件的原样摘录。

```cpp
        } else {
            for (int dimension = 0; dimension < head_size; ++dimension) {
                const std::size_t offset = cache_offset(
                    physical_block, layer_index, head, page_offset,
                    dimension, num_layers, num_heads, head_size);
                dot += to_float(q[query_base + dimension]) *
                    to_float(k_cache[offset]);
            }
        }
        const float score = dot * scale;
        scores[token] = score;
        local_max = fmaxf(local_max, score);
    }
```

这里容易按熟悉的高性能 Attention 实现想错：当前实现的 QK 阶段是**线程按历史 Token 分工，每个线程串行遍历该 Token 的 D 个维度**。

如果 `context_length=17`，前 17 个线程各算一个点积；如果长度 300，线程 0 处理历史位置 0、128、256。它不是一个 warp 合作算一个点积，也没有在这一阶段使用 Tensor Core。

每个历史位置都先经过 `token/16 → block_table → physical_block`。位置 0—15 可以位于物理页 5，位置 16 位于物理页 2；逻辑时间顺序不要求物理地址连续。

与 PyTorch 对照：

```python
# 单行、单头的教学等价式，K_logical 已按页表恢复逻辑顺序。
scores = (K_logical[:context_length] * q[None, :]).sum(dim=-1)
scores = scores / (head_size ** 0.5)
```

共享内存 `scores` 保存整个可见长度的分数；`local_max` 只保存本线程所处理位置的最大值。没有分到位置的线程保留 `-FLT_MAX`，参与 max reduction 时不会压过正常分数。

## 6. 两次归约与同步：逐条解释 barrier

源码：[mini_vllm/cuda/paged_attention.cu，第 176—206 行](../mini_vllm/cuda/paged_attention.cu#L176)。以下为当前文件的原样摘录。

```cpp
    reduction[thread] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] =
                fmaxf(reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }
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

把这段拆成四步：

1. 每个线程把自己的局部最大值写入 `reduction[thread]`，然后同步，保证所有槽已就绪。
2. 步长 64、32、16、8、4、2、1 合并，最终 `reduction[0]` 是全块最大值。
3. 所有线程读取最大值后，额外同步，再复用 `reduction` 做求和。
4. 求指数、求和、求倒数，确保后面的 V 阶段可以读取全部已写好的 `scores`。

为什么读完 maximum 后还有一个同步？不同 warp 的推进速度可以不同。若一个 warp 已开始把局部和写到 `reduction[0]`，另一个 warp 还没读取最大值，就会读到错误含义的数据。

这是**共享内存生命周期**问题：同一块内存在两个阶段表达不同值。仅仅“每轮归约里已经有 barrier”不足以证明跨阶段复用安全。

数值上，`exp(score - maximum)` 避免直接对大分数求指数溢出；最后乘 `inverse_sum` 才得到归一化权重。`scores` 此时存的是未除以总和的指数，不是完整概率。

## 7. V 阶段为什么换一种线程分工

请继续看 [标量 V 加权部分](../mini_vllm/cuda/paged_attention.cu#L238)。这里每个线程拥有一个输出维度，再遍历所有历史 Token：

```cpp
// 教学简化，省略真实 cache_offset 参数；不是可独立编译的源码。
for (int d = thread; d < D; d += blockDim.x) {
    float sum = 0;
    for (int t = 0; t < context_length; ++t) {
        int page = table[t / 16];
        sum += scores[t] * inverse_sum * V[page, t % 16, d];
    }
    out[d] = cast_to_storage_dtype(sum);
}
```

QK 阶段按时间位置分工，V 阶段按特征维度分工。这种实现方便学习，也意味着短上下文、小 head size 时线程利用率可能有限。

FP16 的 `half2` 分支一次读写两个相邻维度，但仍把数值转换成两个 float 累加。它不等于“所有计算都用 half”，也不等于使用 Tensor Core。

## 8. 手工还原一次 Attention

只看一个头，设 `D=2`，Query 为 `[1,0]`，两个可见 K 为 `[1,0]` 与 `[0,1]`，V 为 `[2,0]` 与 `[0,4]`。第二个 KV 可以位于任意合法物理页，逻辑结果不应改变。

```text
scores = [1/sqrt(2), 0]
减去最大值 = [0, -1/sqrt(2)]
exp ≈ [1, 0.493069]
概率 ≈ [0.669762, 0.330238]
out ≈ [1.339523, 1.320954]
```

做两个独立检查：先检查恢复出的逻辑 K/V 与原始张量一致，再比较 Attention 输出。只检查最终输出，可能因为偶然抵消而漏掉某些页写入错误。

仓库提供 CPU PyTorch 演示，适合先建立数值直觉：

```bash
conda run -p /home/miniconda3/envs/zyf1 python \
  doc/from_pytorch/examples/attention_and_pages.py pages
```

该示例为方便手算使用页大小 4；本 CUDA kernel 的页大小固定为 16。概念一致，地址计算时必须使用各自实际常量。

## 9. 独立测试究竟覆盖什么

源码：[dev/cuda/test_paged_attention.cu，第 132—151 行](../dev/cuda/test_paged_attention.cu#L132)。以下为当前文件的原样摘录。

```cpp
        constexpr int batch_size = 3;
        constexpr int num_layers = 2;
        constexpr int layer_index = 1;
        constexpr int num_heads = 4;
        constexpr int head_size = 64;
        constexpr int max_context = 64;
        constexpr int max_blocks =
            max_context / kPagedAttentionPageSize;
        constexpr int num_pages = batch_size * max_blocks;

        const std::vector<int> block_tables = {
            11, 2, 7, 0,
            5, 9, 1, 10,
            3, 8, 4, 6,
        };
        const std::array<std::array<int, batch_size>, 3> cases = {{
            {{1, 15, 16}},
            {{17, 31, 32}},
            {{33, 64, 7}},
        }};
```

页表故意打乱，例如第一行从物理页 11 跳到 2，防止一个把页号当逻辑块号的错误实现侥幸通过。

长度组跨越 15/16/17、31/32/33，还包括容量上界 64。每组长度验证的重点不同：

| 情况 | 容易暴露的问题 |
| --- | --- |
| 长度 1 | 是否包含当前 Token、空历史处理 |
| 15/16 | 页尾位置与整数除法 |
| 17 | 新页写入与第二张页表项 |
| 33 | 多次跨页，不能只支持两页 |
| 64 | 最大上下文与最后合法地址 |
| 不同层 `layer_index=1` | 层 stride 是否正确 |

测试分别对照 dense reference 和 KV 写入结果。`cudaGetLastError` 只能立即发现 launch 配置等错误，异步执行时的非法访存还需要同步或 Compute Sanitizer 才能可靠暴露。

调试时先固定一个请求、一层、一个头，打印 host 上的页表、context、slot；确认地址正确后再看线程归约。否则上游地址错误与下游数值错误混在一起很难定位。

## 10. 性能边界与练习答案

这份 kernel 的目标是验证分页存储和端到端生成，不能据此声称实现了 FlashAttention。它为每个线程块分配与最大上下文成正比的 scores 共享内存，且 QK 的 head 维度由线程串行遍历。

若增加最大上下文，即使本次实际很短，launch 请求的动态共享内存也可能变大。应同时检查设备可用共享内存、launch 是否成功、占用率与实际长度；不能只改一个配置值就假设支持任意长上下文。

**题 1：logical token=32，页表 `[7,1,9]`，页内偏移和物理页是什么？**

答案：偏移 0，物理页 9；当前写槽为 144。不要用 `32*D` 直接访问物理池。

**题 2：为什么 context=17 时不能读取第二页全部 16 个位置？**

答案：只有第二页的第 0 个位置有效，其余可能未写入或属于旧请求。可见长度限制的是逻辑 Token，不是已分配页数。

**题 3：把 maximum 后的额外 barrier 删除，单次测试通过就能证明安全吗？**

答案：不能。warp 调度时序变化可能才触发竞争，应从读写依赖证明同步必要性，并结合工具检查。

**题 4：只交换物理页编号并相应搬动 KV，输出应该改变吗？**

答案：不应改变。页表负责恢复逻辑顺序，这也是随机物理映射测试应验证的不变量。

**题 5：为什么一个 dtype 为 FP16 的 Attention 仍使用 float scores？**

答案：存储精度和归约精度可以不同。点积、指数和求和对误差敏感，当前实现用 FP32 累加后再把最终输出转回存储类型。

下一篇：[任务 04：把这些 kernel 组织成完整 GPU 模型](task_04_gpu_model_runner_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[分页与 Packed 输入](from_pytorch/03_pages_and_packed.md)。
下文记录任务 03 当时的独立 FP32 基线；当前 Kernel 已接入 Packed Prefill 和低精度路径，
写 KV 使用 slot_mapping。请结合任务 05/06 与当前源码阅读。

**状态：已完成独立 FP32 Kernel 基线。**

## 任务目标

这一阶段把 CPU PagedAttention 的核心计算搬到 GPU，并保留 vLLM 风格的分页接口。
实现范围是独立的 FP32 Decode Kernel：输入当前 Token 的 Q/K/V、请求页表和上下文长度，
先把新 K/V 写入缓存，再让 Q 读取该请求的全部历史 K/V。它尚未接入完整 GPT-2
ModelRunner，因此本任务报告的是 Kernel 结果，不是模型端到端吞吐。后续任务 04 已完成
模型接入，端到端结果见 `task_04_gpu_model_runner_zh.md`。

本任务新增约 881 行 C++/CUDA：核心接口和 Kernel 218 行，独立正确性测试 319 行，
Benchmark 344 行。真正需要在面试中讲清楚的是 218 行核心实现；测试和 Benchmark
占多数，因为性能项目必须证明结果正确且可复现。

## 数据布局

```text
Q / new K / new V / output
  [batch, num_heads, head_size]

K Cache / V Cache
  [num_pages, num_layers, num_heads, page_size=16, head_size]

Block Table
  [batch, max_blocks_per_sequence]

Context Length
  [batch]
```

逻辑 Token 位置 `t` 的地址转换为：

```text
logical_page  = t / 16
offset_in_page = t % 16
physical_page = block_table[request][logical_page]
```

Attention 只依赖页表，不要求一个请求的物理页连续。这正是分页 KV Cache 与普通连续
Tensor 的关键区别。

## Kernel 如何工作

入口 `paged_attention_decode()` 连续启动两个 Kernel：

1. `write_kv_cache_kernel`：按最后一个 Token 的逻辑位置查页表，将当前 K/V 写入物理页。
2. `paged_attention_kernel`：一个 CUDA Block 负责一个 `(request, head)`。

第二个 Kernel 使用 128 个线程，执行四个阶段：

```text
每个线程计算若干 Q·K 分数
          ↓
共享内存归约得到最大值
          ↓
exp(score - max)，再归约得到分母
          ↓
每个线程负责若干 Head Dimension，聚合 softmax(score)·V
```

减去最大值后再计算指数，可避免 Softmax 溢出。分数保存在动态共享内存，归约暂存放在
固定共享内存。Q、K、V、页表和上下文长度在调用期间全部位于 GPU。

## 为什么同步点必须认真处理

共享数组 `reduction[]` 先保存局部最大值，随后又被复用来保存局部指数和。每个线程
读取最终最大值后，必须执行一次 `__syncthreads()`，才能允许其他线程覆盖该数组。
缺少这个屏障时，普通正确性测试可能恰好通过，但 `compute-sanitizer --tool racecheck`
会报告读写冲突。

这个问题说明 CUDA 正确性不能只看数值输出。同步错误具有时序依赖，至少要同时做：

- 独立参考实现的数值对齐；
- `memcheck` 检查越界和非法显存访问；
- `racecheck` 检查共享内存数据竞争。

## 正确性验证

测试配置为 Batch 3、2 Layers、4 Heads、Head Size 64，并故意使用乱序物理页。三组
Context Length 为 `{1,15,16}`、`{17,31,32}`、`{33,64,7}`，因此同时覆盖：

- 长度小于一页、正好一页和刚跨页；
- 31/32/33 的第二个页边界；
- 一个 Batch 内不同上下文长度；
- 新 K/V 写入以及历史 K/V 读取；
- 非连续、乱序的 Block Table。

独立 CPU Reference 使用 double 计算稠密 Attention，GPU 输出结果为：

```text
max_abs_error=4.47035e-08
max_rel_error=0.000356036
max_kv_write_error=0
memcheck: 0 errors
racecheck: 0 hazards
```

运行方法：

```bash
make GPU_COMPUTE_CAPABILITY=86 test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 ./test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck \
  ./test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck \
  ./test_cuda_paged_attention
```

## 性能测试

测试设备为 NVIDIA GeForce RTX 3090，Compute Capability 8.6，编译目标 `sm_86`，
CUDA Runtime 12.5。
固定 12 Heads、Head Size 64、Page Size 16。每个配置预热 20 次；每组连续执行 50 次，
重复 20 组，使用 CUDA Event 计量两个 Kernel 的 GPU 时间。

| Batch | Context | P50 | P95 | 算法有效带宽 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 9.196 us | 9.217 us | 12.027 GB/s |
| 1 | 64 | 13.537 us | 13.558 us | 29.955 GB/s |
| 1 | 256 | 32.768 us | 32.788 us | 48.375 GB/s |
| 1 | 512 | 60.119 us | 60.151 us | 52.529 GB/s |
| 8 | 16 | 9.564 us | 9.586 us | 92.505 GB/s |
| 8 | 64 | 15.503 us | 15.525 us | 209.247 GB/s |
| 8 | 256 | 48.701 us | 48.908 us | 260.387 GB/s |
| 8 | 512 | 90.624 us | 90.686 us | 278.780 GB/s |
| 32 | 16 | 9.994 us | 10.015 us | 354.098 GB/s |
| 32 | 64 | 26.604 us | 26.708 us | 487.760 GB/s |
| 32 | 256 | 82.125 us | 82.209 us | 617.656 GB/s |
| 32 | 512 | 190.945 us | 191.995 us | 529.243 GB/s |

“算法有效带宽”按本算法必须读取或写入的 Q/K/V、输出和新 K/V 字节数除以时间计算，
用于比较同一实现的不同形状。它不是 Nsight Compute 硬件计数器测得的实际 DRAM
流量，缓存命中和重复读取都会使两者不同。

原始结果位于：

- `benchmark/results/cuda_paged_attention_rtx3090.json`
- `benchmark/results/cuda_paged_attention_rtx3090.csv`

复现命令：

```bash
make GPU_COMPUTE_CAPABILITY=86 benchmark_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 ./benchmark_cuda_paged_attention \
  --json benchmark/results/cuda_paged_attention_rtx3090.json \
  --csv benchmark/results/cuda_paged_attention_rtx3090.csv
```

## 怎样理解结果

Batch 1、短 Context 时只有 12 个 CUDA Blocks，GPU 并行度不足，启动开销占比很高。
Batch 增大后，同时执行的 `(request, head)` 增多，有效带宽显著提高。Context 从 256
增加到 512 后，B=32 的有效带宽反而下降，说明当前实现的标量内层循环、重复加载 Q、
共享内存分数数组和线程分工仍有优化空间。

本版本的主要限制是：

- 只支持 FP32；
- 每个 `(request, head)` 固定使用一个 Block；
- Q 在每个 Token 的 dot-product 中重复读取；
- 没有使用 `float4`、Warp Shuffle、Tensor Core 或在线 Softmax 融合；
- 只完成独立 Kernel，尚未形成 GPU 端到端模型推理。

这些限制不是需要隐藏的缺点，而是下一轮优化所需的基线。

## 下一步开发任务

下一阶段先做系统接入，再做算子优化：

1. 为 `GPT2ModelRunner` 增加 GPU KV Cache 所有权和生命周期。
2. 将 Block Table、Context Length、Slot Mapping 持久化到设备侧，按调度增量更新。
3. 接通每层 Q/K/V 输出和本 Kernel，构建 GPU Decode 执行路径。
4. 与 CPU 路径比较完整 logits 和生成 Token，并建立端到端 TTFT/TPOT Benchmark。
5. 在正确链路上增加 FP16、向量化加载、Warp Reduction 和融合版本，逐项比较收益。

面试时可以先讲清楚三个问题：页表怎样把逻辑 Token 映射到物理地址；一个
`(request, head)` 内怎样完成稳定 Softmax；为什么 Racecheck 能发现数值测试没有暴露的
共享内存同步错误。能把这三点讲透，比只背一个性能数字更有说服力。
