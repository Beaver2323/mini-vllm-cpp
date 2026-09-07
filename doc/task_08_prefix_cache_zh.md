# 开发任务 08 学习手册：完整 Block Prefix Cache

这份文档按一次真实请求的调用顺序讲 Prefix Cache。阅读时不要只看 `BlockManager`，因为
缓存命中只有同时改变 Sequence 计算进度、Scheduler 调度量、ModelInput 页表和 GPU KV
读取地址，才算真正接通。

## 1. 学习目标与代码导航

学完后应能回答：

1. PagedAttention 与 Prefix Cache 分别解决什么问题？
2. 为什么 Cache Key 不能只包含当前 Block 的 Token？
3. Cache 引用和 Sequence 引用为什么必须分开记账？
4. 命中后为什么修改 `num_computed_tokens` 就能让 Scheduler 跳过前缀？
5. 为什么最后一个 Prompt Block 不缓存？这和 Copy-on-Write 有什么关系？
6. 怎样证明 GPU 读取了旧请求写入的 KV，而不只是复用了 Block ID？

| 学习点 | 实现位置 | 上层调用或验证位置 |
| --- | --- | --- |
| Sequence 计算进度 | [`Sequence`](../mini_vllm/sequence.hpp#L21-L84) | `pending_tokens`、`mark_computed`、`append_token` |
| Prefix Cache 开关 | [`BlockManager` 构造函数](../mini_vllm/block_manager.hpp#L22-L36) | [`GPT2CudaEngine` 构造函数](../mini_vllm/cuda/gpt2_cuda_engine.hpp#L28-L44) |
| Cache 数据结构 | [`Block` 与 `CacheEntry`](../mini_vllm/block_manager.hpp#L17-L20) | 成员变量见 282--289 行 |
| 内容 Key | [`prefix_key`](../mini_vllm/block_manager.hpp#L250-L260) | `apply` 与 `cache` 都调用它 |
| 新请求查缓存 | [`apply_prefix_cache`](../mini_vllm/block_manager.hpp#L101-L131) | [`Scheduler::try_schedule`](../mini_vllm/scheduler.hpp#L123-L144) |
| 新 KV 注册缓存 | [`cache_computed_prefix_blocks`](../mini_vllm/block_manager.hpp#L133-L159) | [`Scheduler::commit`](../mini_vllm/scheduler.hpp#L87-L120) |
| 容量保证与驱逐 | [`ensure_capacity`](../mini_vllm/block_manager.hpp#L69-L96) | [`evict_one_cached_block`](../mini_vllm/block_manager.hpp#L262-L280) |
| 请求完成释放 | [`release`](../mini_vllm/block_manager.hpp#L174-L201) | `Scheduler::commit` 111 行 |
| 页表变成 GPU 地址 | [`prepare_packed_model_input`](../mini_vllm/model_input.hpp#L41-L110) | [`paged_attention_decode`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1034-L1044) |
| 控制面测试 | [`test_prefix_cache_hit_and_lru_eviction`](../dev/test_mini_vllm_control_plane.cpp#L120-L168) | Block Size 4，便于手算 |
| GPU 模型级测试 | [`test_gpt2_cuda_prefix_cache.cu`](../dev/cuda/test_gpt2_cuda_prefix_cache.cu#L34-L116) | FP16，真实 12 层 KV 复用 |

## 2. 先区分 PagedAttention 与 Prefix Cache

PagedAttention 让一个请求的逻辑 Token 可以放到不连续物理 KV Block：

```text
sequence.block_table = [7, 2, 11]
逻辑 Token 0..15   → 物理 Block 7
逻辑 Token 16..31  → 物理 Block 2
逻辑 Token 32..47  → 物理 Block 11
```

它解决存储组织，但新请求仍会重新 Prefill。Prefix Cache 再允许两个请求的 Block Table
引用同一个已经计算好的物理 Block：

```text
请求 A：[公共前缀 0..15] [A 后缀]
          block_table[0] = 3
                              ┌─ 同一个 GPU KV Block 3
请求 B：[公共前缀 0..15] [B 后缀]
          block_table[0] = 3
```

Prefix Cache 的收益是跳过公共前缀的模型计算；PagedAttention 提供让这个共享关系可表达的
页表和物理内存基础。

## 3. 从 Engine 看完整命中调用链

开关从
[`GPT2CudaEngine` 构造函数](../mini_vllm/cuda/gpt2_cuda_engine.hpp#L28-L44)
传给 BlockManager：

```cpp
GPT2CudaEngine(..., bool enable_prefix_cache = false)
    : block_manager_(
          num_kv_blocks, kPagedAttentionPageSize,
          enable_prefix_cache),
      scheduler_(scheduler_config, block_manager_),
      model_runner_(..., block_manager_, ...) {}
```

Scheduler 和 ModelRunner 持有同一个 `block_manager_` 的引用，所以 Scheduler 写入的
`sequence.block_table()` 会被 ModelRunner 直接读取。

一次 `step()` 的真实顺序在
[`gpt2_cuda_engine.hpp` 75--97 行](../mini_vllm/cuda/gpt2_cuda_engine.hpp#L75-L97)：

```cpp
SchedulerOutput output = scheduler_.schedule();
result.sampled_token_ids = model_runner_.run(output);
scheduler_.commit(output, result.sampled_token_ids);
```

Prefix Cache 穿过这三个阶段：

```text
schedule
  └─ Waiting 请求先 apply_prefix_cache
       ├─ block_table 接上缓存物理 Block
       └─ num_computed_tokens 前移
  └─ ensure_capacity 只为剩余 Token 分配页

model_runner.run
  └─ 从前移后的 position 开始构造 ModelInput
  └─ GPU PagedAttention 根据共享 block_table 读取旧 KV

commit
  └─ mark_computed(本轮实际执行 Token 数)
  └─ cache_computed_prefix_blocks 注册新完成的完整 Prompt Block
  └─ 请求结束时 release Sequence 引用
```

## 4. Sequence 中哪个变量让调度真正跳过前缀

先看
[`Sequence`](../mini_vllm/sequence.hpp#L39-L63)：

```cpp
std::size_t num_computed_tokens() const { return num_computed_tokens_; }

std::size_t pending_tokens() const {
    return token_ids_.size() - num_computed_tokens_;
}

void mark_computed(std::size_t count) {
    if (count > pending_tokens()) throw ...;
    num_computed_tokens_ += count;
}
```

假设 Prompt 长 18，命中 1 个 16-Token Block：

```text
命中前：num_computed_tokens = 0，pending_tokens = 18
命中后：num_computed_tokens = 16，pending_tokens = 2
```

Scheduler 使用 `pending_tokens()` 计算本轮数量，ModelInput 使用
`num_computed_tokens() + offset` 计算绝对位置。因此只要命中函数同时接好页表并调用
`mark_computed(16)`，整个下游会自然从位置 16 开始。

## 5. Cache 数据结构与所有权

### 5.1 物理 Block 只有一个引用计数

定义在
[`block_manager.hpp` 17--20 行](../mini_vllm/block_manager.hpp#L17-L20)：

```cpp
struct Block {
    int id = -1;
    std::size_t ref_count = 0;
};
```

`ref_count` 的来源有两类，但存放在同一个计数器：

| 状态 | `ref_count` | 是否能进 Free List | 是否可被 Cache LRU 驱逐 |
| --- | ---: | --- | --- |
| 完全空闲 | 0 | 是 | 不在 Cache 中 |
| 只有 Prefix Cache 持有 | 1 | 否 | 是 |
| Cache + 一个活跃 Sequence | 2 | 否 | 否 |
| Cache + 两个活跃 Sequence | 3 | 否 | 否 |

### 5.2 Cache Entry 保存什么

定义在 245--248 行，容器在 286--289 行：

```cpp
struct CacheEntry {
    int block_id = -1;
    std::uint64_t last_used = 0;
};

std::map<std::vector<int>, CacheEntry> prefix_cache_;
std::uint64_t cache_clock_ = 0;
std::size_t prefix_cache_hit_blocks_ = 0;
```

`block_id` 指向真正存储 K/V 的物理页；`last_used` 用于 LRU；Map 的 Key 是 Token 内容。
`prefix_cache_hit_blocks_` 是累计命中计数，不是当前缓存大小。当前大小由
`prefix_cache_.size()` 返回。

## 6. 为什么 Key 必须包含完整历史

实现位于
[`prefix_key`](../mini_vllm/block_manager.hpp#L250-L260)：

```cpp
const std::size_t end = (logical_block + 1) * block_size_;
return std::vector<int>(
    sequence.token_ids().begin(),
    sequence.token_ids().begin() + end);
```

Block Size 为 4 时：

```text
Prompt = [A B C D | E F G H | I]
Key(block 0) = [A B C D]
Key(block 1) = [A B C D E F G H]
```

不能让 `Key(block 1) = [E F G H]`。Transformer 的 K/V 由当前 Token 表示产生，而当前表示
已经通过前层 Attention 融入前文。即使第二块 Token 都是 `[E F G H]`，前面分别是
`[A B C D]` 和 `[W X Y Z]` 时，第二块的 K/V 通常不同。

教学版使用完整 `vector<int>` 作为 Key，语义最直观。生产系统通常组合前一块哈希与当前块
Token 得到链式哈希，避免 Key 长度随逻辑块增长。

## 7. 新请求怎样连续命中缓存

实现位置：
[`apply_prefix_cache`](../mini_vllm/block_manager.hpp#L101-L131)。

### 7.1 只允许全新 Waiting Sequence 应用一次

```cpp
if (!prefix_cache_enabled_ || sequence.num_computed_tokens() != 0 ||
    !sequence.block_table().empty()) {
    return 0;
}
```

这一条件防止同一个请求在 Chunked Prefill 的后续轮次重复增加引用计数。调用点也只在
[`Scheduler::try_schedule`](../mini_vllm/scheduler.hpp#L123-L127) 的 Waiting 分支：

```cpp
if (sequence->status() == SequenceStatus::Waiting) {
    block_manager_.apply_prefix_cache(*sequence);
}
```

请求成功接纳后状态变成 Running，后续调度不再查 Prefix Cache。

### 7.2 只复用连续前缀

```cpp
for (std::size_t logical_block = 0;
     logical_block < cacheable_blocks; ++logical_block) {
    const std::vector<int> key = prefix_key(sequence, logical_block);
    auto cached = prefix_cache_.find(key);
    if (cached == prefix_cache_.end()) break;
    ...
}
```

第一个 Miss 就 `break`。因为后面的 Block Key 虽然理论上也能查到，但前一段上下文缺失时
不能直接跳过去，Sequence 的 `num_computed_tokens` 也只能表示一个连续已计算前缀。

### 7.3 命中时完成三件事

```cpp
++block.ref_count;                         // 新 Sequence 引用
cached->second.last_used = ++cache_clock_; // 更新 LRU
sequence.block_table().push_back(block.id);// 接上物理页
...
sequence.mark_computed(hits * block_size_);// 跳过计算
prefix_cache_hit_blocks_ += hits;          // 统计
```

少做其中任何一件都会出问题：不增加引用会让活跃页被驱逐；不接页表会让 PagedAttention
找不到旧 KV；不前移计算进度会重新 Prefill 并覆盖共享页。

## 8. 新计算完成的 Block 怎样进入 Cache

调用点在
[`Scheduler::commit`](../mini_vllm/scheduler.hpp#L87-L113)：

```cpp
sequence.mark_computed(item.num_scheduled_tokens);
block_manager_.cache_computed_prefix_blocks(sequence);
```

先 `mark_computed`，再注册缓存，因为缓存函数只允许已经完整算完的 Block。

实现位置：
[`cache_computed_prefix_blocks`](../mini_vllm/block_manager.hpp#L133-L159)：

```cpp
const std::size_t prompt_cacheable =
    (sequence.num_prompt_tokens() - 1) / block_size_;
const std::size_t computed_blocks =
    sequence.num_computed_tokens() / block_size_;
const std::size_t count = std::min(
    {prompt_cacheable, computed_blocks, sequence.block_table().size()});
```

这三个上限分别保证：只缓存允许的 Prompt Block、只缓存已经完成计算的 Block、只访问已经
存在的页表项。对一个新 Key：

```cpp
++block.ref_count; // Prefix Cache 自身持有一个引用
prefix_cache_.emplace(
    std::move(key), CacheEntry{block_id, ++cache_clock_});
```

随后请求若结束，`release(sequence)` 只减掉 Sequence 引用，Cache 引用仍让 K/V 留在原物理页。

## 9. 为什么故意留下最后一个 Prompt Block

两处都使用：

```cpp
cacheable_blocks = (num_prompt_tokens - 1) / block_size;
```

示例，Block Size 16：

| Prompt 长度 | 可缓存完整 Block | 下个相同请求至少重算 |
| ---: | ---: | ---: |
| 15 | 0 | 15 Token |
| 16 | 0 | 16 Token |
| 17 | 1 | 1 Token |
| 32 | 1 | 16 Token |
| 33 | 2 | 1 Token |

为什么 16 Token Prompt 不直接命中全部 16？自回归生成需要最后一个 Prompt 位置的 logits。
如果 16 个 Token 全部跳过，当前 Runner 没有缓存该位置的 logits，也没有 Token 可送入模型。

更关键的是共享页写入边界。若新请求命中一个未满的 Partial Block，随后把新 K/V 写到同一页，
就会修改其他 Sequence 共享的物理页。生产实现需要 Copy-on-Write：写入前发现引用数大于 1，
复制该页再继续写。当前版本只共享完整块并至少重算尾部，从设计上避开 COW，便于先掌握
Prefix Cache 的主路径。

## 10. LRU 驱逐和原子容量保证

`Scheduler::try_schedule` 在确定本轮 Token 数后调用
[`ensure_capacity`](../mini_vllm/block_manager.hpp#L69-L96)：

```cpp
const std::size_t additional =
    required - sequence.block_table().size();
while (additional > free_block_ids_.size() &&
       evict_one_cached_block()) {}
if (additional > free_block_ids_.size()) return false;
```

先尝试驱逐出足够页，确认容量后才进入分配循环，因此不会只给 Sequence 分一半页就失败。

LRU 实现在
[`evict_one_cached_block`](../mini_vllm/block_manager.hpp#L262-L280)：

```cpp
if (block.ref_count == 1 &&
    (victim == prefix_cache_.end() ||
     entry.last_used < victim->second.last_used)) {
    victim = it;
}
```

只有 `ref_count == 1` 才说明该页仅被 Cache 持有，可以驱逐。若为 2 或更大，还有活跃
Sequence 正在通过 Block Table 读取它，绝不能回收到 Free List。

选中 Victim 后：

```cpp
--block.ref_count;                 // 去掉 Cache 引用：1 → 0
free_block_ids_.push_back(block.id);
prefix_cache_.erase(victim);
```

显式清空缓存的
[`clear_prefix_cache`](../mini_vllm/block_manager.hpp#L161-L172)
使用同样的引用释放规则；若页仍被 Sequence 使用，只去掉 Cache 引用，不加入 Free List。

## 11. Cache 命中怎样变成 GPU KV 复用

这是最容易只看控制面而漏掉的一段。

命中后，`prepare_packed_model_input` 从前移后的计算位置开始，位置在
[`model_input.hpp` 62--100 行](../mini_vllm/model_input.hpp#L62-L100)：

```cpp
const std::size_t position =
    sequence.num_computed_tokens() + offset;
const int physical_block =
    block_manager.block_id_for_token(sequence, position);

input.slot_mapping.push_back(
    physical_block * block_size + position % block_size);
std::copy(sequence.block_table().begin(),
          sequence.block_table().end(),
          input.block_tables.begin() + row_start);
```

对于 18 Token 的第二请求，`num_computed_tokens=16`，所以只构造 position 16 和 17 两行。
但每行完整复制 Sequence 的 Block Table，其中第 0 项仍指向第一请求留下的物理页。

Runner 把 `block_tables`、`context_lengths` 和 `slot_mapping` 上传 GPU，并在每一层调用
[`paged_attention_decode`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1034-L1044)：

```cpp
paged_attention_decode(
    query, key, value,
    key_cache, value_cache,
    block_tables, context_lengths, slot_mapping,
    attention, batch_size, ..., layer, ...);
```

GPU KV Cache 是 ModelRunner 生命周期内持久存在的 `key_cache_` 和 `value_cache_`。第一请求
完成后只释放 BlockManager 的 Sequence 引用，没有清零物理页。因此第二请求通过相同物理
Block ID 读取的是第一请求在 12 层中真正写入的 K/V。

## 12. 两层测试分别证明什么

### 12.1 控制面测试：引用计数与 LRU

入口：
[`test_prefix_cache_hit_and_lru_eviction`](../dev/test_mini_vllm_control_plane.cpp#L120-L168)。

Block Size 4，总共 4 个物理 Block：

```cpp
BlockManager manager(4, 4, /*enable_prefix_cache=*/true);
Scheduler scheduler({1, 16}, manager);
```

第一请求 `[1..8, 90]` 计算 9 Token，缓存前两个完整 Block。第二请求 `[1..8, 91]`：

```cpp
assert(second->num_computed_tokens() == 8);
assert(second_step.items[0].num_scheduled_tokens == 1);
assert(second->block_table()[0] == cached_block_ids[0]);
assert(second->block_table()[1] == cached_block_ids[1]);
```

随后 13 Token 的不共享请求需要 4 页，而 Free List 只有 2 页，迫使 LRU 驱逐两个只由
Cache 持有的页。这个测试不运行模型，专门验证状态机和所有权。

### 12.2 GPU 测试：真实 KV 内容复用

入口：
[`dev/cuda/test_gpt2_cuda_prefix_cache.cu`](../dev/cuda/test_gpt2_cuda_prefix_cache.cu#L34-L116)。

关键构造：

```cpp
std::vector<int> second_prompt(
    first_prompt.begin(), first_prompt.begin() + 16);
second_prompt.push_back(1234);
second_prompt.push_back(4321);
```

第一个 Prompt 长 17，留下一个 16 Token Cache Block；第二个 Prompt 长 18，前 16 相同：

```cpp
const CudaEngineStepResult second_step = engine.step();
assert(second_step.num_batched_tokens == 2);
assert(engine.prefix_cache_hit_blocks() == 1);
```

测试随后比较第二请求最后位置的完整词表 GPU logits 与 CPU 完整前缀 Reference，并检查
Greedy Token 一致。这证明被复用的是 12 层 FP16 K/V 内容，不只是控制面计数。

```text
cached_blocks=1
hit_blocks=1
second_prompt_tokens=18
second_scheduled_tokens=2
max_abs_logit_error=0.104279
generated_token_equal_cpu=true
```

## 13. 复现与调试命令

```bash
cd /home/users/zyf/zyf_llm.c/llm.c

make test_minivllm_control_plane
./test_minivllm_control_plane

make GPU_COMPUTE_CAPABILITY=86 test_gpt2_cuda_prefix_cache
CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_prefix_cache
```

建议按顺序设置断点或日志：

1. `Scheduler::try_schedule`：确认只对 Waiting 请求调用 Apply。
2. `BlockManager::apply_prefix_cache`：观察 Key、Hit、Ref Count 和 Block Table。
3. `Scheduler::try_schedule` 的 `count`：确认 18 变成 2。
4. `prepare_packed_model_input`：确认 position 从 16 开始。
5. `paged_attention_decode`：确认 Block Table 第一项仍是缓存页。
6. `Scheduler::commit`：观察缓存注册早于 Sequence Release。
7. `evict_one_cached_block`：确认只选择 `ref_count == 1` 的 Victim。

## 14. 当前边界与下一步设计题

当前实现有意保留以下限制：

- Key 使用完整 Token 向量，长 Prompt 下的存储与比较成本较高；
- 只复用从逻辑 Block 0 开始的连续完整前缀；
- 最后一个 Prompt Block 总会重新计算；
- 没有 Partial Block Cache 和 Copy-on-Write；
- Cache 只存在于单个 Engine 生命周期内；
- 没有多模型、多进程或分布式 Cache 隔离。

建议亲手完成三个练习：

1. 用 Block Size 4 手算 `[1..8, 90]` 和 `[1..8, 91]` 的 Key、引用计数与 Free List 变化。
2. 在 `apply_prefix_cache` 打印每次 Key 长度和命中 Block ID，运行控制面测试核对推导。
3. 设计 Partial Block Cache：写出何时触发 COW、复制哪些层的 K/V、怎样更新 Block Table，
   先写设计说明，不立即编码。

完成后应能解释：Prefix Cache 不是一个 Token 到 Block ID 的普通 Map。它是内容 Key、
Sequence 计算进度、物理页共享、引用计数、调度跳过和 LRU 驱逐共同组成的生命周期协议。
