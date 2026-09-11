# Mini-vLLM C++ 源码学习手册

**PyTorch 开发者零基础阅读：先完成 [新入门路线](from_pytorch/README.md) 第 1—3 节，再回来按本手册定位实现。**
入门路线补齐自回归生成、KV 因果性、请求生命周期，提供 PyTorch CPU 实验和 nano-vLLM 对照。

本文是整个项目的源码索引，负责回答“先学什么、去哪里看、从哪里调用、怎样验证”。每个
专项文档继续解释实现细节。不要一次把所有文件通读；按本文的阶段完成代码跟踪和练习。

## 逐任务源码精读怎么使用

任务 01—11 已逐篇扩写，打开原文件即可看到新的源码精读。每篇先讲当前实现，再保留
原开发记录与历史实验；不用在另一份总览中寻找遗漏的关键代码。

这一轮精读核对的源码基线是 `59027d7`。源码摘录标明文件和行号，点击可定位；行号随
后续开发可能变化，函数名和调用链是更稳定的阅读入口。标注“教学伪代码”的代码框用于
说明语义，不是项目提供的 Python API，也不保证可以直接编译。

| 任务 | 本篇重点推导 | 读完后的检查 |
| --- | --- | --- |
| [01 请求到模型](task_01_gpt2_model_runner_zh.md) | Engine 三阶段、CPU 微步、采样资格、Workspace 生命周期 | 手推 Prompt 5、预算 3、生成 3 的四轮计数 |
| [02 Benchmark](task_02_benchmark_zh.md) | 工作量、计时边界、首 Token 观察点、分位数插值 | 重算 344/92 输入数及 TTFT/TPOT |
| [03 CUDA Attention](task_03_cuda_paged_attention_zh.md) | 五维地址、线程分工、两次归约、共享内存同步 | 算出页内地址并解释每个 barrier |
| [04 GPU Runner](task_04_gpu_model_runner_zh.md) | 设备缓冲、stream、cuBLAS 布局、层内残差数据流 | 从 PyTorch Linear 对应到 GEMM 参数 |
| [05 Packed Prefill](task_05_multi_token_prefill_zh.md) | token budget、请求边界、逐行元数据、因果可见性 | 手写 Decode + Prefill 混合输入表 |
| [06 混合精度](task_06_mixed_precision_zh.md) | 权重转换、存储/累加 dtype、half2、显存公式 | 标出一次前向中的舍入位置 |
| [07 融合与 Graph](task_07_fusion_cuda_graph_zh.md) | 双输出、跨层 LN、Graph key、捕获与资源生命周期 | 解释 `(N,R)` 相同但行索引不同为何能重放 |
| [08 Prefix Cache](task_08_prefix_cache_zh.md) | 完整历史 key、引用所有权、LRU、分配失败副作用 | 手推 A/B 共享一页到 clear 的全部 ref 变化 |
| [09 采样行裁剪](task_09_sample_rows_zh.md) | 资格判断、Gather、紧凑输出还原、R=0 | 对齐 packed 行、logits 行和请求索引 |
| [10 缓存测量](task_10_prefix_benchmark_zh.md) | seed 构造、off/miss/hit、计数差值、CSV 分析 | 算出四种前缀下输入数并解释计时成本 |
| [11 双卡 PD](task_11_pd_disaggregation_zh.md) | 首 Token 交接、页号重映射、host staging、并发与背压 | 复述 17 Token 请求迁移后的第一步 Decode |

建议每次只读一篇的 2—3 节：先读代码框并找到调用方，再遮住答案做手算，最后使用已有
测试或 CPU 演示检查预测。篇幅增加是为了支持逐段学习，不要求一次读完所有功能。

## 1. 先建立全局图

项目分成控制面和执行面：

```text
用户 add_request
  └─ Sequence：Token、状态、计算进度、Block Table
       └─ Scheduler::schedule：选择请求和本轮 Token 数
            └─ BlockManager：查 Prefix Cache、分配/共享物理页
                 └─ ModelInput：Token、Position、Context、Slot、Block Table
                      └─ GPT2CudaModelRunner::run
                           ├─ Embedding / cuBLAS GEMM / LayerNorm / MLP
                           ├─ CUDA PagedAttention：写新 KV、按页表读历史 KV
                           ├─ Fusion / CUDA Graph
                           └─ GPU Argmax
                                └─ Scheduler::commit
                                     ├─ 前移 computed tokens
                                     ├─ 注册 Prefix Cache
                                     ├─ 追加生成 Token
                                     └─ 完成请求并释放 Block
```

总入口在
[`GPT2CudaEngine::step`](../mini_vllm/cuda/gpt2_cuda_engine.hpp#L75-L98)：

```cpp
SchedulerOutput output = scheduler_.schedule();
result.sampled_token_ids = model_runner_.run(output);
scheduler_.commit(output, result.sampled_token_ids);
```

第一次阅读只需记住这三步。后续每个模块都能放回这条链路中。

## 2. 完整学习路线

| 阶段 | 需要掌握 | 主要代码 | 专项文档 | 完成标准 |
| ---: | --- | --- | --- | --- |
| 1 | Sequence 状态和 Token 记账 | `mini_vllm/sequence.hpp` | [任务 01](task_01_gpt2_model_runner_zh.md) | 手算 Prefill/Decode 的三个计数 |
| 2 | Block Pool 和逻辑页表 | `mini_vllm/block_manager.hpp`、`paged_kv_cache.hpp` | [分页基础](#4-阶段二blockmanager-与分页地址) | 手算 Token 到物理 KV 地址 |
| 3 | Token Budget 与 Continuous Batching | `mini_vllm/scheduler.hpp` | [任务 01](task_01_gpt2_model_runner_zh.md) | 手推三个请求的调度轨迹 |
| 4 | Scheduler 到模型的五类元数据 | `mini_vllm/model_input.hpp` | [任务 05](task_05_multi_token_prefill_zh.md) | 解释每个数组由谁生产、谁消费 |
| 5 | CPU Reference 和增量推理 | `mini_vllm/gpt2_model_runner.hpp`、`train_gpt2.cpp` | [任务 01](task_01_gpt2_model_runner_zh.md) | 解释旧 KV 为什么可复用 |
| 6 | CUDA PagedAttention | `mini_vllm/cuda/paged_attention.cu` | [任务 03](task_03_cuda_paged_attention_zh.md) | 能讲清 Grid、页表寻址和 Softmax |
| 7 | GPU Runner 与 Packed Prefill | `mini_vllm/cuda/gpt2_cuda_model_runner.cu` | [任务 04](task_04_gpu_model_runner_zh.md)、[任务 05](task_05_multi_token_prefill_zh.md) | 从 Schedule 跟踪到 Argmax |
| 8 | FP16/BF16 与 Tensor Core | 同上 `matmul`、低精度 Kernel | [任务 06](task_06_mixed_precision_zh.md) | 解释存储精度和累加精度 |
| 9 | Fusion 与 CUDA Graph | 同上 `forward<T>` | [任务 07 详细手册](task_07_fusion_cuda_graph_zh.md) | 解释固定地址与动态数据 |
| 10 | Prefix Cache | `block_manager.hpp`、`scheduler.hpp` | [任务 08 详细手册](task_08_prefix_cache_zh.md) | 跟踪命中、共享、释放和 LRU |
| 11 | Benchmark 与性能证据 | `benchmark/`、`benchmark/results/` | [任务 02](task_02_benchmark_zh.md) | 区分 TTFT、TPOT、吞吐和初始化成本 |

新增任务按下面顺序继续，不必同时学习：

| 任务 | 先看代码 | 再看文档 | 完成标准 |
| --- | --- | --- | --- |
| 09 采样行裁剪 | Runner::run、gather_sample_rows_kernel、graph_key | [逐函数讲解](task_09_sample_rows_zh.md) | 手算 N、R，解释为何需要二维 Graph Key |
| 10 前缀性能对照 | benchmark_gpt2_cuda_prefix_cache.cu 的 measure/main | [计时与调用链](task_10_prefix_benchmark_zh.md) | 看懂 off/miss/hit 的计数和 TTFT |
| 11 双 GPU PD | gpt2_pd_engine.hpp 的 step/try_handoff，再看 Runner::copy_kv_to | [请求交接与 KV 迁移](task_11_pd_disaggregation_zh.md) | 手推 Prompt 17 Token 的交接及 D 第一步 |

三项的 [实测数据、正确性与复现命令](../benchmark/results/task09_11/README.md) 独立保存。

建议每天只完成一个阶段。先读“主要代码”，再运行指定测试，最后不看文档复述调用链。

## 3. 阶段一：Sequence 是请求状态的唯一来源

### 代码位置

- 数据结构：[`mini_vllm/sequence.hpp:12--84`](../mini_vllm/sequence.hpp#L12-L84)
- 构造入口：[`GPT2CudaEngine::add_request`](../mini_vllm/cuda/gpt2_cuda_engine.hpp#L46-L73)
- 状态消费者：[`Scheduler::schedule`](../mini_vllm/scheduler.hpp#L56-L83)
- 状态更新：[`Scheduler::commit`](../mini_vllm/scheduler.hpp#L87-L120)

### 三个最重要的计数

```cpp
std::size_t num_tokens() const;           // 当前已有 Prompt + 生成 Token
std::size_t num_prompt_tokens() const;    // 构造后固定
std::size_t num_computed_tokens() const;  // 已经执行模型并写好 KV 的 Token

std::size_t pending_tokens() const {
    return token_ids_.size() - num_computed_tokens_;
}
```

例：Prompt 长 5，第一次 Prefill 完成并采样 Token 99 后：

```text
刚创建：num_tokens=5, num_prompt_tokens=5, num_computed_tokens=0
Prefill Commit：先 computed=5，再 append 99，所以 num_tokens=6
下一轮 Decode：pending_tokens=6-5=1
```

`append_token` 要求 `num_computed_tokens == token_ids.size()`，确保只有当前所有输入都计算完
才能追加采样结果。这里是状态不变量，建议在
[`append_token`](../mini_vllm/sequence.hpp#L59-L64) 下断点观察。

### 必须回答

- Prefill 和 Decode 是否由两个 Sequence 类表示？不是，由 `is_prefill()` 根据计数判断。
- 新生成 Token 何时写 KV？采样后只追加到 Sequence，下一轮 Decode 执行时才写 KV。

## 4. 阶段二：BlockManager 与分页地址

### 代码位置

- 控制面物理块：[`BlockManager`](../mini_vllm/block_manager.hpp#L22-L96)
- Sequence 页表：[`Sequence::block_table`](../mini_vllm/sequence.hpp#L73-L83)
- CPU KV 数据池：[`KVCachePool`](../paged_kv_cache.hpp#L15-L66)
- CPU PagedAttention：[`paged_attention_forward`](../paged_kv_cache.hpp#L85-L175)
- 分配/释放测试：[`test_block_allocation_release_and_reuse`](../dev/test_mini_vllm_control_plane.cpp#L19-L45)

`BlockManager` 管“谁拥有哪一页”，`KVCachePool` 或 CUDA Runner 管“页里面的 K/V 数据”。
两者通过相同的物理 Block ID 对接。

```cpp
const std::size_t logical_block = token_index / block_size_;
const int physical_block = sequence.block_table()[logical_block];
const std::size_t page_offset = token_index % block_size_;
```

缓存布局是：

```text
[physical_block, layer, head, page_offset, head_dimension]
```

元素偏移：

```text
offset = ((((p * L + l) * H + h) * P + o) * D + d)
```

示例：Page Size 16，`block_table=[5,2,9]`。Token 20 的逻辑页为 1、页内偏移为 4，所以
访问物理页 2。分页的“不连续”是 Block ID 不连续，底层大池仍可一次连续分配。

### 容量保证调用点

Scheduler 在
[`try_schedule`](../mini_vllm/scheduler.hpp#L131-L143) 中计算目标 Token 数：

```cpp
const std::size_t count = std::min(sequence->pending_tokens(), budget);
const std::size_t target = sequence->num_computed_tokens() + count;
if (!block_manager_.ensure_capacity(*sequence, target)) return false;
```

`ensure_capacity` 先确认可用页数量，再统一分配，因此 OOM 不会留下半张 Block Table。

## 5. 阶段三：Scheduler 与 Continuous Batching

### 代码位置

- 输出协议：[`ScheduledItem`、`SchedulerOutput`](../mini_vllm/scheduler.hpp#L21-L30)
- 调度主函数：[`Scheduler::schedule`](../mini_vllm/scheduler.hpp#L56-L83)
- 单请求调度：[`Scheduler::try_schedule`](../mini_vllm/scheduler.hpp#L123-L144)
- 状态提交：[`Scheduler::commit`](../mini_vllm/scheduler.hpp#L87-L120)
- 测试：[`test_chunked_prefill`](../dev/test_mini_vllm_control_plane.cpp#L47-L68)、[`test_continuous_admission_and_retirement`](../dev/test_mini_vllm_control_plane.cpp#L90-L118)

调度顺序先 Running、后 Waiting：

```cpp
for (const auto& sequence : running_) {
    try_schedule(sequence, output);
}
while (!waiting_.empty() && budget_remains) {
    if (!try_schedule(waiting_.front(), output)) break;
    // Waiting → Running
}
```

这让已接纳的请求先于新 Waiting 请求推进；Running 也可能包含未完成的 Prefill，
因此当前实现并不是严格的 Decode 优先策略。剩余 Token Budget 用于准入新请求。`count = min(pending_tokens, budget)` 让长 Prompt 被拆成多个 Chunk。

### 手算练习

配置 `max_num_sequences=2, max_num_batched_tokens=5`，请求 A Prompt 5、请求 B Prompt 2。
第一轮 A 使用全部 5 Token；下一轮 A 有 1 个 Decode Token，B 可用剩余 4 Token 中的 2 个，
于是同一轮出现 Decode + Prefill。对应断言在测试 100--110 行。

## 6. 阶段四：ModelInput 是控制面和 CUDA 的契约

### 代码位置

- 结构定义：[`ModelInput`](../mini_vllm/model_input.hpp#L15-L31)
- Packed 构造：[`prepare_packed_model_input`](../mini_vllm/model_input.hpp#L41-L110)
- GPU 调用：[`GPT2CudaModelRunner::Impl::run`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L740-L798)
- GPU H2D：[`forward<T>`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1079-L1099)

| 数组 | 生产方式 | GPU 消费位置 | 含义 |
| --- | --- | --- | --- |
| `token_ids` | Sequence 当前待算 Token | Embedding Kernel | 查 Token Embedding |
| `positions` | `computed + offset` | Embedding Kernel | 查绝对位置编码 |
| `context_lengths` | `position + 1` | PagedAttention | 当前 Q 可见多少历史 Token |
| `slot_mapping` | 物理块 × 页大小 + 页内偏移 | Write KV Kernel | 新 K/V 写到哪里 |
| `block_tables` | 复制 Sequence 页表 | Attention Kernel | 历史逻辑 Token 在哪个物理页 |
| `query_start_locations` | 每请求在 Packed Batch 的边界 | Runner 采样映射 | 哪一行属于哪个请求 |

核心地址构造位于 62--100 行：

```cpp
const std::size_t position = sequence.num_computed_tokens() + offset;
const int physical_block =
    block_manager.block_id_for_token(sequence, position);
const std::size_t physical_slot =
    physical_block * block_manager.block_size() +
    block_manager.slot_for_token(position);
```

这是面试中最值得手画的数据流：Scheduler 不传裸 K/V 指针，只传逻辑进度和 Block Table，
ModelInput 把它们转成设备 Kernel 所需的紧凑元数据。

## 7. 阶段五：CPU 路径是正确性 Reference

### 代码位置

- CPU Runner：[`mini_vllm/gpt2_model_runner.hpp`](../mini_vllm/gpt2_model_runner.hpp#L19-L149)
- 推理 Workspace：[`train_gpt2.cpp:38`](../train_gpt2.cpp#L38)
- 增量模型前向：[`gpt2_forward_inference_with_workspace`](../train_gpt2.cpp#L389)
- CPU PagedAttention：[`paged_kv_cache.hpp:85`](../paged_kv_cache.hpp#L85)
- 模型级测试：[`dev/test_gpt2_engine.cpp`](../dev/test_gpt2_engine.cpp)

CPU Runner 为 Chunked Prefill 建立多个单 Token 微批次：

```cpp
for (std::size_t micro_step = 0; micro_step < max_micro_steps; ++micro_step) {
    ModelInput input = prepare_model_input(output, micro_step, ...);
    gpt2_forward_inference_with_workspace(
        &model_, input.token_ids.data(), &kv_cache_pool_, &page_table,
        input.batch_size(), &workspace_);
}
```

它速度不是最终目标，作用是把调度、分页和增量推理逻辑做成易调试基线。GPU 测试用完整
前缀 CPU Forward 比较全词表 logits，避免 CUDA 实现自己和自己对比。

旧 Token 的 K/V 可以复用，因为因果注意力中位置 `i` 看不到未来 Token；新增 Token 不会
改变 `i` 已经算出的 K/V。旧 Q 不需要缓存，因为下一步只使用新 Token 的 Q 查询历史 K/V。

## 8. 阶段六：CUDA PagedAttention

### 代码位置

- 接口与 Page Size：[`paged_attention.cuh`](../mini_vllm/cuda/paged_attention.cuh#L11-L52)
- 写 KV：[`write_kv_cache_kernel`](../mini_vllm/cuda/paged_attention.cu#L57-L104)
- Attention：[`paged_attention_kernel`](../mini_vllm/cuda/paged_attention.cu#L106-L254)
- Launch 封装：[`paged_attention_decode_impl`](../mini_vllm/cuda/paged_attention.cu#L256-L292)
- 独立测试：[`dev/cuda/test_paged_attention.cu`](../dev/cuda/test_paged_attention.cu)

Launch Grid 是：

```cpp
const dim3 grid(batch_size, num_heads);
```

所以一个 CUDA Block 处理一个 Packed Token 的一个 Attention Head。先启动
`write_kv_cache_kernel` 写本轮 K/V，再启动 `paged_attention_kernel` 读取从位置 0 到
`context_length-1` 的所有 K/V。

写地址来自 `slot_mapping`：

```cpp
const int physical_slot = slot_mapping[request];
const int physical_block = physical_slot / kPagedAttentionPageSize;
const int page_offset = physical_slot % kPagedAttentionPageSize;
```

读历史地址来自 `block_tables`：

```cpp
const int physical_block = block_tables[
    request * max_blocks_per_sequence +
    token / kPagedAttentionPageSize];
const int page_offset = token % kPagedAttentionPageSize;
```

两套元数据作用不同：Slot Mapping 是本轮单点写位置；Block Table 是整个历史读取映射。
Softmax 使用减最大值保证稳定性，低精度存储路径仍转 FP32 累加。

## 9. 阶段七：GPU Runner 与 Packed Prefill

### 代码位置

- 配置/接口：[`gpt2_cuda_model_runner.cuh`](../mini_vllm/cuda/gpt2_cuda_model_runner.cuh#L13-L58)
- 设备 Buffer 生命周期：[`gpt2_cuda_model_runner.cu:650--709`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L676-L738)
- Runner 上层入口：[`Impl::run`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L740-L798)
- Transformer 前向：[`forward<T>`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1079-L1353)
- 端到端测试：[`test_gpt2_cuda_model_runner.cu`](../dev/cuda/test_gpt2_cuda_model_runner.cu)

GPU Runner 不为每步重新 `cudaMalloc`。权重、KV Cache、中间激活和 logits 在构造时按最大
容量分配并保持地址稳定。每轮只上传 ModelInput 元数据，最后只把 Argmax Token 拷回 CPU。

核心层循环按以下次序读：

```text
Embedding
for each layer:
  LayerNorm1
  QKV GEMM + Split QKV
  Write KV + PagedAttention
  Attention Projection + Residual
  LayerNorm2
  FC GEMM + GELU + Projection + Residual
Final LayerNorm + Vocabulary GEMM + Argmax
```

Packed Prefill 把多个请求本轮的所有 Token 作为 GEMM Batch 维度。每个 Token 仍有自己的
Position、Context Length、Slot 和 Block Table，所以合并 GEMM 不会破坏因果性。

## 10. 阶段八：FP16/BF16、Tensor Core 和数值边界

专项阅读：[任务 06：混合精度](task_06_mixed_precision_zh.md)。

### 代码位置

- 精度枚举：[`CudaDataType`](../mini_vllm/cuda/gpt2_cuda_model_runner.cuh#L13-L17)
- 普通 GEMM：[`matmul`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L986-L1024)
- 词表 GEMM：[`logits_matmul`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1026-L1050)
- `half2` PagedAttention：[`paged_attention.cu:76--93`](../mini_vllm/cuda/paged_attention.cu#L76-L93)

低精度路径使用 FP16/BF16 存权重、激活和 KV Cache；LayerNorm 均值/方差、Attention Dot、
Softmax 和 Value 聚合用 FP32 累加。`cublasGemmEx` 指定 `CUBLAS_COMPUTE_32F` 和
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`，让存储精度与累加精度分离。

当前正式推荐 FP16。BF16 已接通，但 GPT-2 测试存在 Argmax 分歧，因此文档和简历不能把
BF16 写成与 FP16 同等级的正式性能结论。

## 11. 阶段九：Fusion 与 CUDA Graph

详细阅读：[任务 07：Fusion 与 CUDA Graph 逐函数手册](task_07_fusion_cuda_graph_zh.md)。

### 最短调用路径

```text
GPT2CudaConfig.enable_fused_residual_layernorm / enable_cuda_graph
  └─ Impl::forward<T>
       ├─ copy_metadata（Graph 外）
       ├─ cuda_graphs_.find({total_tokens, num_logit_rows})
       ├─ Replay，或 Capture 整段计算
       └─ fused_residual_layernorm（Attention 后、MLP 后）
```

关键代码：

```cpp
copy_metadata(...);                       // 更新固定地址里的动态数据
const auto graph_key = std::make_pair(batch_size, num_logit_rows);
const auto graph = cuda_graphs_.find(graph_key);
if (graph != cuda_graphs_.end()) {
    cudaGraphLaunch(graph->second.executable, stream_.get());
}
```

Graph 固定设备地址和执行拓扑，数据由 Replay 前的 H2D 更新。Fusion 单独使用是负优化；
四组 A/B 中 Graph 是主要收益。不要用“Kernel 少了”直接推出“更快”。

## 12. 阶段十：Prefix Cache

详细阅读：[任务 08：Prefix Cache 逐函数手册](task_08_prefix_cache_zh.md)。

### 最短调用路径

```text
Scheduler::try_schedule(Waiting)
  └─ BlockManager::apply_prefix_cache
       ├─ 查完整历史 Token Key
       ├─ 增加物理页引用
       ├─ 写 Sequence Block Table
       └─ mark_computed(命中页数 × Page Size)
            └─ Scheduler 只调度剩余 Token

Scheduler::commit
  └─ cache_computed_prefix_blocks
       └─ Cache 持有额外引用
```

核心代码：

```cpp
++block.ref_count;
sequence.block_table().push_back(block.id);
sequence.mark_computed(hits * block_size_);
```

三行分别建立所有权、物理映射和计算跳过。缺任何一个都不构成正确的 Prefix Cache。

## 13. 阶段十一：测试与 Benchmark 去哪里看

| 你要验证的能力 | 测试/Benchmark | 关键观察值 |
| --- | --- | --- |
| Block 分配、OOM、调度、Prefix LRU | `dev/test_mini_vllm_control_plane.cpp` | Block Table、Free Count、Ref Count |
| CPU PagedAttention 数学正确性 | `dev/test_paged_attention_resume.cpp` | 与独立 Dense Reference 误差 |
| CPU Engine 闭环 | `dev/test_gpt2_engine.cpp` | 混合 Prefill/Decode、Token 一致 |
| CUDA PagedAttention | `dev/cuda/test_paged_attention.cu` | 跨页/乱序页、memcheck/racecheck |
| GPU Runner | `dev/cuda/test_gpt2_cuda_model_runner.cu` | 全词表 logits、Argmax、Graph 数 |
| GPU Prefix Cache | `dev/cuda/test_gpt2_cuda_prefix_cache.cu` | 18 Token 只调度 2 Token |
| CPU 服务指标 | `benchmark/benchmark_gpt2_serving.cpp` | TTFT、TPOT、吞吐 |
| CUDA 服务与 A/B | `benchmark/benchmark_gpt2_cuda_serving.cu` | Fusion/Graph 四组合 |

构建和运行：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c

make test_minivllm_control_plane test_gpt2_engine
./test_minivllm_control_plane
OMP_NUM_THREADS=16 ./test_gpt2_engine

make GPU_COMPUTE_CAPABILITY=86 \
  test_cuda_paged_attention test_gpt2_cuda_model_runner \
  test_gpt2_cuda_prefix_cache benchmark_gpt2_cuda_serving

CUDA_VISIBLE_DEVICES=0 ./test_cuda_paged_attention
CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_model_runner --precision fp16 --cuda-graph
CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_prefix_cache
```

## 14. 性能结果应怎样解释

固定 RTX 3090、4 请求、Prompt 8/16/24/32、每请求生成 4 Token：

| 阶段 | 吞吐 | 相对上一关键基线 | 主要变化 |
| --- | ---: | ---: | --- |
| CUDA 逐 Token Prefill | 295.708 tok/s | 基线 | 每个 Token 单独模型调用 |
| Packed Prefill FP32 | 1838.148 tok/s | 6.2× | 合并 GEMM，Launch 数下降 |
| Packed Prefill FP16 | 2738.953 tok/s | +46.8% vs 同版 FP32 | Tensor Core、低精度存储 |
| FP16 + CUDA Graph | 3068.092 tok/s | +15.9% vs FP16 Eager | 降低 CPU Launch 开销 |
| FP16 + Fusion + Graph | 3189.767 tok/s | +20.5% vs FP16 Eager | Graph 加融合组合 |

这张表汇集各开发阶段的历史实验，部分阶段同时改变了实现与测量环境，不能把每一行都看成
只改一个变量的因果对照。具体 A/B 以对应任务原始数据为准。
这些数字只对应固定测试负载，不代表所有 Batch、Prompt 和 GPU。面试时应同时说清硬件、
精度、请求形状、Warmup 和比较基线。

## 15. 学习时的断点清单

按一次 18 Token、命中 16 Token Prefix 的请求设置：

1. `GPT2CudaEngine::add_request`：看初始 Sequence。
2. `Scheduler::try_schedule`：进入 Waiting 分支。
3. `BlockManager::apply_prefix_cache`：看 Hit 和 Ref Count。
4. `Scheduler::try_schedule` 的 `count`：确认只剩 2。
5. `prepare_packed_model_input`：确认 positions 是 16、17。
6. `GPT2CudaModelRunner::Impl::run`：确认 Packed Batch Size 为 2。
7. `forward<T>` 的 `copy_metadata`：确认五类 H2D 数据。
8. `cuda_graphs_.find`：看对应 Shape 的 Capture 或 Replay。
9. `paged_attention_decode`：看共享 Block Table 进入 GPU。
10. `Scheduler::commit`：看 computed、Cache 注册、Token 追加和 Release。

如果 CUDA Kernel 不能直接断点，先在 Host 调用点打印元数据，再使用 Compute Sanitizer 或
Nsight Systems 验证设备执行。

## 16. 面试前必须能独立回答

1. KV Cache 为什么减少计算？为什么不缓存 Q？
2. PagedAttention 为什么改善内存管理，但不保证单 Kernel 更快？
3. `slot_mapping` 和 `block_tables` 有何区别？
4. Chunked Prefill 怎样与 Decode 混在一个调度轮？
5. Packed Prefill 为什么仍保持因果性？
6. 为什么 FP16 存储仍用 FP32 做 LayerNorm 和 Softmax 归约？
7. CUDA Graph 固定地址后，动态请求如何更新？
8. Fusion 为什么会负优化？怎样设计 A/B 才能发现？
9. Prefix Cache Key 为什么包含完整历史？
10. 为什么 Cache-only 页可驱逐，活跃共享页不可驱逐？
11. 为什么当前实现不缓存最后一个 Prompt Block？
12. GPU Prefix Cache 测试怎样证明复用了真实 KV 数据？

## 17. 推荐的实际学习节奏

如果此前没接触过推理引擎，先读 [从 PyTorch 出发的第 1—3 节](from_pytorch/README.md)，
运行小模型缓存与真实控制面 demo，再开始下述源码遍历。

第一遍只读 `Sequence → Scheduler → BlockManager → Engine::step`，运行控制面测试。第二遍加
`ModelInput → CPU Runner → PagedAttention`，手算一条页表。第三遍进入 CUDA Runner，先跟
Packed Prefill，再跟混合精度。第四遍只学习任务 07 的 Fusion/Graph。第五遍只学习任务 08
的 Prefix Cache。

每一遍都输出三样东西：一张调用图、一个手算例子、一次测试结果。能在不看文档时把这三样
复述出来，才算真正掌握；不用一次记住所有 Kernel 细节。
