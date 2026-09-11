# 第 3 节：从连续 Tensor 到分页 KV，再到 Packed 输入

上一节：[请求与调度](02_requests_and_scheduler.md) · [目录](README.md) · 下一节：[PyTorch 到 CUDA](04_pytorch_to_cuda.md)

本节沿两条数据链阅读：历史 K/V 存在哪里，以及本轮新 Token 怎样找到这些数据。

## 1. 如果用一个普通 Tensor 保存缓存

最容易写出的缓存是每请求、每层一个 `[H,max_length,D]` Tensor，再用切片写入新位置。
它能增量推理，但需要提前决定容量：预留太大会浪费，容量小了又可能需要重新分配和复制。
多请求长短不同、不断到达结束时，固定预留尤其不灵活。

分页把容量拆成固定 Token 数的块，只在需要时分配相应块。**分页不是量化或压缩**：相同 dtype
和已缓存 Token 数下，每个 Token 的 K/V 字节数不变。它改变容量分配和共享粒度。

也不要把它等同于 PyTorch CUDA caching allocator：后者管理 Tensor 的底层内存分配，项目的
BlockManager 管一个预先分配好的 KV Pool 内部，哪些页分给哪个请求。

## 2. 四种“块/位置”分别是什么

| 名称 | 本节含义 | 示例 |
| --- | --- | --- |
| Token Position | 请求内部的绝对逻辑位置 | 第 9 个输入的位置为 8 |
| Logical Block | 请求内部按 P 个 Token 分组的页序号 | P=4 时位置 8 属于逻辑页 2 |
| Physical Block | KV Pool 里的页编号 | 逻辑页 2 分配到物理页 4 |
| CUDA Thread Block | Kernel 中协作执行的一组线程 | 与 KV 页号没有必然对应关系 |

本节“物理页”只是 KV Pool 的应用级逻辑存储单元，不是显卡 MMU 的硬件页。

设 P=4，页表 `block_table=[3,0,4]`：

```text
请求位置 0..3  → 逻辑页 0 → Pool 物理页 3
请求位置 4..7  → 逻辑页 1 → Pool 物理页 0
请求位置 8..11 → 逻辑页 2 → Pool 物理页 4
```

位置 8 的 Slot 为 `4*4+0=16`。Slot 表示池中的 Token 槽，不是字节偏移；还需要指定层、头、
维度和 dtype 才能定位具体 K/V 元素。

## 3. 用 PyTorch 索引理解真实布局

CUDA Runner 的 K 与 V 各自分配一块连续池，视为：

```text
[num_pages, num_layers, num_heads, page_size, head_dim]
```

某个元素的 Python 解释：

```python
value = k_pool[physical_page, layer, head, offset_in_page, dimension]
```

C++ 指针需要自己计算展平后的元素索引：

```cpp
((((page * num_layers + layer) * num_heads + head) * page_size + offset) * head_dim + dim)
```

乘 `sizeof(T)` 才是字节偏移，C++ 的 `T* + index` 已隐含按元素大小移动。
真实地址函数见 [cache_offset](../../mini_vllm/cuda/paged_attention.cu#L17)。Pool 可以一次
`cudaMalloc` 连续分配；某请求的 Block ID 序列仍可以不连续。

## 4. 管理“页号”与存储“数据”是两份职责

调用路径如下：

```text
Scheduler::try_schedule
  └─ BlockManager::ensure_capacity → Sequence.block_table
       └─ prepare_packed_model_input → slot_mapping / block_tables
            └─ Runner::forward
                 └─ paged_attention_decode
                      ├─ write_kv_cache_kernel
                      └─ paged_attention_kernel
```

| 要追踪的内容 | 实现 | 调用点 |
| --- | --- | --- |
| 分配页号 | [ensure_capacity](../../mini_vllm/block_manager.hpp#L69) | Scheduler::try_schedule |
| 计算新 KV 的 Slot | [prepare_packed_model_input](../../mini_vllm/model_input.hpp#L41) | Runner::run |
| 分配 KV 数据池 | [Runner::Impl 构造](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu)，搜索 `key_cache_(cache_elements()` | Engine 构造 Runner |
| 写本轮 K/V | [write_kv_cache_kernel](../../mini_vllm/cuda/paged_attention.cu#L57) | paged_attention_decode_impl |
| 读历史 K/V | 同文件 `paged_attention_kernel` | 写入 Kernel 之后 |
| 释放页号 | [BlockManager::release](../../mini_vllm/block_manager.hpp#L174) | Scheduler::commit 结束分支 |

释放 Block 不等于 cudaFree。大池仍由 Runner 持有，BlockManager 归还页号，后续请求覆盖写入。
因此看到进程显存没有立刻下降，不代表这里发生 KV 页泄漏；先看 free_blocks 和引用计数。

## 5. 亲自把不连续页还原成 Attention

```bash
python doc/from_pytorch/examples/attention_and_pages.py pages
```

读 [pages_lesson](examples/attention_and_pages.py)：它先生成连续参考 K/V，再按 `[3,0,4]` 放进
Pool。未使用的槽故意填 NaN，用来检查是否错误读取了尾页。

```python
for position in range(length):
    page, offset = table[position // page_size], position % page_size
    k_pool[page, :, offset] = k[:, position]
    v_pool[page, :, offset] = v[:, position]
```

实验为了直观看懂，先用 `torch.stack` 把逻辑历史 Gather 回连续 Tensor，再计算 Attention。
项目 CUDA Kernel 则在遍历历史时直接按 Block Table 取数据，没有先把全部历史重排为一个
连续大 Tensor。两者数学相同，访存实现和性能不同。

## 6. Packed 输入为什么不是普通的 [B,T]

A 已有 5 Token KV，本轮输入一个 Decode Token 17；B 新到达，Prompt 为 `[19,23,29]`。
如果用填充后的矩阵，需要为 A 补齐到与 B 一样的本轮长度。项目改成一维压紧输入：

```text
原始请求：A = [17]，B = [19,23,29]
Packed：  [17,19,23,29]
请求数 B=2，输入行数 N=4
Embedding 后：[N,C]
```

Linear/LayerNorm/MLP 可以对这些行统一执行，不需要知道它们属于哪个请求。但 Attention
必须保持每个请求独立的上下文，因此需要旁边的元数据，而不是把四行当成同一段文本。

## 7. 一组元数据完整手算

仍用教学页大小 4；A 页表 `[3,1]`，B 页表 `[4]`。A 的旧 KV 已存在，B 尚无旧 KV。

| Packed 行 | 请求 | Token | 绝对 Position | Context Length | 新 KV Slot | 请求页表 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | A | 17 | 5 | 6 | 5 | `[3,1]` |
| 1 | B | 19 | 0 | 1 | 16 | `[4]` |
| 2 | B | 23 | 1 | 2 | 17 | `[4]` |
| 3 | B | 29 | 2 | 3 | 18 | `[4]` |

其他边界与映射：

```text
token_ids             = [17,19,23,29]
positions             = [5,0,1,2]
context_lengths       = [6,1,2,3]
slot_mapping          = [5,16,17,18]
query_start_locations = [0,1,4]
scheduled_item_indices= [0,1,1,1]
sample_rows           = [0,3]
```

`query_start_locations` 把每个请求的行区间表示成前缀和，A 是 `[0,1)`，B 是 `[1,4)`。
`sample_rows` 是需要预测下一 Token 的最后一行，不能用请求下标 `[0,1]` 代替。

本项目为了简化 Kernel，把同一请求的 Block Table 复制到它的每个 Packed 行，并填充到固定
`max_blocks_per_sequence`。所以第 1、2、3 行的页表相同。这是教学实现的空间换简单策略，
并不意味着成熟框架都使用同样的元数据布局。

运行验证：[packed_lesson](examples/attention_and_pages.py)。

```bash
python doc/from_pytorch/examples/attention_and_pages.py packed
```

## 8. Slot Mapping 与 Block Table 为什么都需要

写本轮 K/V 时，每行只写一个新位置，Slot 提供直接目的地：

```cpp
const int physical_slot = slot_mapping[request];
const int physical_block = physical_slot / kPagedAttentionPageSize;
const int page_offset = physical_slot % kPagedAttentionPageSize;
```

这里 Kernel 的 `request` 变量在 Packed 路径中其实是行号。读历史时，一行 Q 需要依次读取
很多 Token，必须每个位置都查页表：

```cpp
const int physical_block = block_tables[
    request * max_blocks_per_sequence + token / kPagedAttentionPageSize];
const int page_offset = token % kPagedAttentionPageSize;
```

因此 Slot 回答“我写哪里”，Block Table 回答“我要读的历史分别在哪里”。只复制 Block Table
而没有保留真实 KV 内容，下一步仍然算不对；任务 11 的跨卡迁移就是同时保持这两者语义。

## 9. 同轮写入很多 Token，为什么不会看到未来

CUDA 先在同一 Stream 上启动写 KV Kernel，把当前层所有新 K/V 写完，再启动 Attention。
同 Stream 的顺序保证第二个 Kernel 能读到前一个的结果。每行 Q 仍按自己的 Context Length
遍历，所以 B 的第一行只读位置 0，不会读到 B 后面两行。

同时，A/B 页表指向各自的缓存，因此不会互相串请求。分页共享只在明确相同前缀时建立，
不是所有请求都能读取池中任意页。

## 10. Prefix Cache 在分页之上增加了什么

同一请求重用自己的历史是 KV Cache；不同请求发现完整前缀相同，重用已计算页，是 Prefix
Cache。只是最后几个 Token 一样不够，因为各层 K/V 依赖之前的上下文。

本项目使用“从 Prompt 开始到该块末尾的完整 Token 向量”作为 Key，固定同一模型；命中完整
页后，增加页引用、填入请求页表，并前移 computed。最后至少留一个 Prompt Token 重算，以
获得生成所需 logits，避免给请求写入共享的部分尾页。

引用计数可这样手算：

```text
请求 A 独占页                         ref=1
Prefix Cache 再持有一个引用            ref=2
A 结束                               ref=1，仅缓存持有
请求 B 命中                          ref=2，B+缓存
B 结束                               ref=1，可被缓存驱逐
```

本项目将缓存自己的引用也计入 ref_count，所以 cache-only=1。不要把它照搬到其他框架：
不同实现可能只统计活跃请求引用。具体区别见第 5 节的 nano-vLLM 对照。

## 11. 过关任务

1. 把页表改成 `[4,2,0]`，重新计算位置 8 的 Slot，运行 pages 实验验证。
2. A 已有 5 Token、B 本轮只执行 Prompt 前 2 个时，sample_rows 应包含哪些行？
3. 为什么 B 的 Context Length 是 1/2/3，而不是都等于 Prompt 总长度 3？
4. 为什么释放请求后 GPU 大池还在，free_blocks 却增加？
5. 解释 Page、Slot、Position、CUDA Thread Block 的区别。

完成后进入第 4 节，把熟悉的 PyTorch 算子与真正的 CUDA 执行点逐个对上。
