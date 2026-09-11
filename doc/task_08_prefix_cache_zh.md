# 开发任务 08 学习手册：完整 Block Prefix Cache

Prefix Cache 的核心不是多一个 map，而是让多个请求共享已经计算好的 KV，同时保证后续写入不会破坏共享数据。本篇用一条完整生命周期解释引用计数、命中、释放和淘汰。

学习目标：在纸上跟踪每个物理页由谁持有，并说明一个新请求为什么能够跳过已命中的 Prompt 计算。

## 1. 与 PyTorch past_key_values 的区别

普通单请求缓存让**同一请求**不重复计算历史。Prefix Cache 进一步让**不同请求**复用相同前缀的 KV。

相同文本经过分词后不一定有相同 ID 序列，反过来仅末尾一段 ID 相同也不代表历史 hidden 相同。这里缓存键直接建立在 Token 前缀上，模型权重和推理配置在 Engine 内保持一致。

当前实现是单 Engine 的内存缓存，不持久化、不跨模型共享，也没有多租户 salt 或 LoRA 身份管理。学习时先证明这个范围内正确，再考虑额外身份字段。

## 2. 源码阅读地图与调用点

| 位置 | 在生命周期中做什么 |
| --- | --- |
| [try_schedule](../mini_vllm/scheduler.hpp#L123) | 新请求先尝试命中 |
| [apply_prefix_cache](../mini_vllm/block_manager.hpp#L101) | 共享页，增加 computed |
| [cache_computed_prefix_blocks](../mini_vllm/block_manager.hpp#L133) | 前向完成后把可缓存页登记 |
| [release](../mini_vllm/block_manager.hpp#L174) | 请求归还自己的引用 |
| [evict_one_cached_block](../mini_vllm/block_manager.hpp#L262) | 回收只有缓存持有的冷页 |
| [clear_prefix_cache](../mini_vllm/block_manager.hpp#L161) | 清除缓存拥有的引用 |
| [GPU 验证](../dev/cuda/test_gpt2_cuda_prefix_cache.cu) | 共享后数值、尾页与回收是否正确 |

```text
新请求 Waiting
  → apply_prefix_cache：已有 KV 的页直接挂到请求表
  → ensure_capacity：为剩余输入申请可写页
  → Runner：只执行 computed 之后的 Token
  → commit：更新 computed → 登记完整前缀页
  → 请求完成：release 请求引用
  → 缓存仍可保留，等命中或 LRU 淘汰
```

## 3. 为什么 key 包含完整历史

源码：[mini_vllm/block_manager.hpp，第 250—260 行](../mini_vllm/block_manager.hpp#L250)。以下为当前文件的原样摘录。

```cpp
    std::vector<int> prefix_key(
        const Sequence& sequence, std::size_t logical_block) const {
        const std::size_t end = (logical_block + 1) * block_size_;
        if (end > sequence.num_prompt_tokens()) {
            throw std::out_of_range("prefix block exceeds prompt");
        }
        return std::vector<int>(
            sequence.token_ids().begin(),
            sequence.token_ids().begin() +
                static_cast<std::ptrdiff_t>(end));
    }
```

对逻辑块 b，key 是 `tokens[0:(b+1)*block_size]`。第二块的 key 不只包含第二块自己的 Token，还包含第一块。

手算反例，教学页大小 4：

```text
请求 A：[1,2,3,4] [9,9,9,9] [100]
请求 B：[5,6,7,8] [9,9,9,9] [100]
```

第二块的 Token 字面相同，但在多层 causal Transformer 中，它们的 hidden 已依赖不同历史，不能共享第二块 KV。完整前缀 key 会把两个请求区分开。

这份实现使用 `map<vector<int>,...>`，容易检查且没有 hash 碰撞歧义，但构造长前缀 key 和比较 vector 有成本。不能把它表述成已经实现了高效的链式哈希缓存。

## 4. 为什么始终留至少一个 Prompt Token 重算

源码：[mini_vllm/block_manager.hpp，第 101—113 行](../mini_vllm/block_manager.hpp#L101)。以下为当前文件的原样摘录。

```cpp
    std::size_t apply_prefix_cache(Sequence& sequence) {
        if (!prefix_cache_enabled_ || sequence.num_computed_tokens() != 0 ||
            !sequence.block_table().empty()) {
            return 0;
        }
        const std::size_t cacheable_blocks =
            (sequence.num_prompt_tokens() - 1) / block_size_;
        std::size_t hits = 0;
        for (std::size_t logical_block = 0;
             logical_block < cacheable_blocks; ++logical_block) {
            const std::vector<int> key = prefix_key(sequence, logical_block);
            auto cached = prefix_cache_.find(key);
            if (cached == prefix_cache_.end()) break;
```

可缓存块数是 `(prompt_tokens−1)/block_size`。减 1 的原因：当前缓存只有 KV，没有保存可直接用于这次生成的最终 logits，Runner 需要至少一行输入产生首 Token。

用真实页大小 16：

| Prompt 长度 | 可命中完整块数上限 | 即使全命中仍需计算 |
| ---: | ---: | ---: |
| 16 | 0 | 16 个 Token |
| 17 | 1 | 1 个 Token |
| 32 | 1 | 16 个 Token |
| 33 | 2 | 1 个 Token |

Prompt 长度正好对齐页大小时，会留下整块作为本请求私有部分。这个策略保守但简单，不需要缓存末行 hidden/logits，也不需要处理共享尾页继续写入的 copy-on-write。

“没有复用所有可能的 Token”是当前设计选择，不是整数除法错误。任务 10 构造 seed 为 `prefix+1`，就是为了让 prefix 个 Token 全部满足可缓存条件。

## 5. 命中时只改元数据，不复制 K/V

源码：[mini_vllm/block_manager.hpp，第 114—130 行](../mini_vllm/block_manager.hpp#L114)。以下为当前文件的原样摘录。

```cpp
            Block& block = blocks_.at(
                static_cast<std::size_t>(cached->second.block_id));
            if (block.ref_count == 0) {
                throw std::logic_error("prefix cache references a free block");
            }
            ++block.ref_count;
            cached->second.last_used = ++cache_clock_;
            sequence.block_table().push_back(block.id);
            ++hits;
        }
        if (hits == 0) return 0;
        if (!live_sequences_.insert(sequence.request_id()).second) {
            throw std::logic_error("cached sequence allocation is inconsistent");
        }
        sequence.mark_computed(hits * block_size_);
        prefix_cache_hit_blocks_ += hits;
        return hits;
```

命中后发生三件关键事情：

1. 物理页引用计数加 1，表示新请求也在使用。
2. 页号加入新请求的 `block_table`，该请求读历史时会指向同一物理位置。
3. `mark_computed(hits*block_size)` 跳过已经存在 KV 的输入位置。

没有调用 memcpy，因为两个请求在同一个 KV Pool 中共享同一个页。省掉的不只是复制，更是整段 Prompt 在所有模型层上的重复计算。

但元数据命中本身不能证明 KV 有效。缓存登记必须发生在模型已完成这些位置计算之后，不能在刚分配页时就登记。

## 6. 登记缓存发生在 commit，缓存自己也持有引用

源码：[mini_vllm/block_manager.hpp，第 133—158 行](../mini_vllm/block_manager.hpp#L133)。以下为当前文件的原样摘录。

```cpp
    void cache_computed_prefix_blocks(const Sequence& sequence) {
        if (!prefix_cache_enabled_) return;
        const std::size_t prompt_cacheable =
            (sequence.num_prompt_tokens() - 1) / block_size_;
        const std::size_t computed_blocks =
            sequence.num_computed_tokens() / block_size_;
        const std::size_t count = std::min(
            {prompt_cacheable, computed_blocks,
             sequence.block_table().size()});
        for (std::size_t logical_block = 0;
             logical_block < count; ++logical_block) {
            const std::vector<int> key = prefix_key(sequence, logical_block);
            auto cached = prefix_cache_.find(key);
            if (cached != prefix_cache_.end()) {
                cached->second.last_used = ++cache_clock_;
                continue;
            }
            const int block_id = sequence.block_table()[logical_block];
            Block& block = blocks_.at(static_cast<std::size_t>(block_id));
            if (block.ref_count == 0) {
                throw std::logic_error("cannot cache an unreferenced block");
            }
            ++block.ref_count; // Prefix Cache 自身持有一个引用。
            prefix_cache_.emplace(
                std::move(key), CacheEntry{block_id, ++cache_clock_});
        }
```

`count` 取三个上界的最小值：Prompt 允许缓存的完整块、实际已计算完整块、已分配页表长度。只分配未计算的页，不能通过 `computed_blocks` 这一关。

如果 key 已存在，只更新访问时间，不再额外增加缓存引用；否则同一个请求每次 commit 都重复加引用，会产生无法回收的泄漏。

本项目的引用规则是：

```text
ref_count = 活跃请求对该页的引用数量 +（缓存登记是否持有 1 个引用）
```

所以空闲页 ref=0；只有缓存持有的页 ref=1；缓存加一个请求通常 ref=2。阅读 nano-vllm 或其他系统时，不要直接套用这些具体数字，不同实现的空闲队列与缓存引用约定可能不同。

## 7. 两个请求共享前缀的完整状态表

教学配置：4 个物理页、页大小 16、Prompt 长度 17、输出 1 个。A 与 B 前 16 个 Token 相同，最后一个 Token 可以不同。

假设 A 先分配页 `[0,1]`，算完 Prompt 后只登记页 0。A 结束时按逆序释放，页 1 回到空闲队列，页 0 留给缓存：

| 时刻 | 页 0 引用组成 | 页 0 ref | 空闲页数 | 缓存页数 |
| --- | --- | ---: | ---: | ---: |
| 初始 | 无 | 0 | 4 | 0 |
| A 分配两页 | A | 1 | 2 | 0 |
| A 计算完成并登记页 0 | A + cache | 2 | 2 | 1 |
| A 完成并释放 | cache | 1 | 3 | 1 |
| B 命中页 0 | cache + B | 2 | 3 | 1 |
| B 分配一个私有尾页 | cache + B | 2 | 2 | 1 |
| B 完成并释放 | cache | 1 | 3 | 1 |
| clear cache | 无 | 0 | 4 | 0 |

B 在命中后 `computed=16`，只需要处理自己的最后一个 Prompt Token。它写私有尾页，绝不会覆盖共享页 0 中的前 16 个位置。

若 A 尚未结束，B 已经命中同一页，页 0 ref 可以是 3：A、B、cache 各一个。它不能被 LRU 淘汰。

## 8. release 与 clear 的区别

源码：[mini_vllm/block_manager.hpp，第 189—201 行](../mini_vllm/block_manager.hpp#L189)。以下为当前文件的原样摘录。

```cpp
        for (auto it = sequence.block_table().rbegin();
             it != sequence.block_table().rend(); ++it) {
            Block& block = blocks_.at(static_cast<std::size_t>(*it));
            if (block.ref_count == 0) {
                throw std::logic_error("block reference count underflow");
            }
            --block.ref_count;
            if (block.ref_count == 0) {
                free_block_ids_.push_back(block.id);
            }
        }
        sequence.block_table().clear();
    }
```

源码：[mini_vllm/block_manager.hpp，第 161—172 行](../mini_vllm/block_manager.hpp#L161)。以下为当前文件的原样摘录。

```cpp
    void clear_prefix_cache() {
        for (const auto& item : prefix_cache_) {
            Block& block = blocks_.at(
                static_cast<std::size_t>(item.second.block_id));
            if (block.ref_count == 0) {
                throw std::logic_error("prefix cache reference underflow");
            }
            --block.ref_count;
            if (block.ref_count == 0) free_block_ids_.push_back(block.id);
        }
        prefix_cache_.clear();
    }
```

`release(sequence)` 归还一个请求拥有的引用，并清空该请求页表；`clear_prefix_cache()` 只归还缓存引用，不清空活跃请求页表。

因此执行 clear 时如果仍有请求使用某缓存页，该页 ref 从 2 降为 1，仍不能进入 free list。直到请求 release 后才归零。

反过来，所有请求完成后 `free<num_blocks` 不一定是泄漏，可能是缓存有意保留。验证时要区分“请求已结束”和“连缓存也已清空”。

`prefix_cache_hit_blocks` 是累计计数，clear 不重置它。测量一次目标请求命中多少块，要用前后差值，任务 10 的 Benchmark 正是这样做。

## 9. LRU 只选 cache-only 页

源码：[mini_vllm/block_manager.hpp，第 262—279 行](../mini_vllm/block_manager.hpp#L262)。以下为当前文件的原样摘录。

```cpp
    bool evict_one_cached_block() {
        auto victim = prefix_cache_.end();
        for (auto it = prefix_cache_.begin(); it != prefix_cache_.end(); ++it) {
            const Block& block = blocks_.at(
                static_cast<std::size_t>(it->second.block_id));
            if (block.ref_count == 1 &&
                (victim == prefix_cache_.end() ||
                 it->second.last_used < victim->second.last_used)) {
                victim = it;
            }
        }
        if (victim == prefix_cache_.end()) return false;
        Block& block = blocks_.at(
            static_cast<std::size_t>(victim->second.block_id));
        --block.ref_count;
        free_block_ids_.push_back(block.id);
        prefix_cache_.erase(victim);
        return true;
```

`ref_count==1` 在遍历缓存条目的前提下意味着只有缓存持有。满足条件后才比较 `last_used`，选择最久未使用的候选。

这里 `last_used` 是单调递增访问计数，不是系统时间。它足以表达相对新旧，不需要为每次 cache hit 读取墙钟。

LRU 不会把正在运行请求的页抢走。本实现没有请求级抢占、换出或自动重算，因此“池满了”可能仍然无法继续，不能把缓存淘汰说成完整的 OOM 恢复机制。

## 10. 分配失败的副作用要讲准确

源码：[mini_vllm/block_manager.hpp，第 69—79 行](../mini_vllm/block_manager.hpp#L69)。以下为当前文件的原样摘录。

```cpp
    bool ensure_capacity(Sequence& sequence, std::size_t token_count) {
        const std::size_t required = blocks_needed(token_count);
        if (required <= sequence.block_table().size()) {
            return true;
        }
        const std::size_t additional = required - sequence.block_table().size();
        while (additional > free_block_ids_.size() &&
               evict_one_cached_block()) {}
        if (additional > free_block_ids_.size()) {
            return false;
        }
```

代码先尝试淘汰缓存，再检查剩余空闲页是否足够。成功检查后才把新页逐个挂到请求表，因此容量不足时不会只分给这个请求一半所需的新页。

但失败不代表全局状态完全没变化：之前的循环可能已经淘汰一些 cache-only 页，只是仍不够满足本次申请。缓存内容与空闲队列可能已改变。

这一区别对写测试很重要。可以要求“请求没有部分新增页”，不能未经源码证明就要求“失败后所有缓存条目原样保留”。

页号合法、ref 与 free list 一致也不是全部正确性；还要证明缓存身份正确、数值已计算、共享范围不被写入。结构检查与数值测试需要配合。

## 11. 调试位置、练习与答案

建议按顺序在 `apply_prefix_cache`、`cache_computed_prefix_blocks`、`release`、`evict_one_cached_block` 停下，观察同一个页号的拥有者变化。不要只打印 ref 数字而不记录对应请求。

用 [GPU Prefix Cache 测试](../dev/cuda/test_gpt2_cuda_prefix_cache.cu) 核对：相同前缀命中、不同历史不能误命中、尾页可写隔离、活跃引用保留，以及输出与独立前向一致。

**题 1：Prompt 48 个 Token，最多可复用几页？**

答案：`(48−1)/16=2` 页，剩下 16 个 Token 重算。不能把全部 3 页都挂上后让 Runner 接收空输入。

**题 2：一页 cache + A + B，A 结束后 ref 是多少？**

答案：2；B 与缓存还各持有一个。此时不能淘汰。

**题 3：调用 clear 后有活跃请求引用的页会立即 free 吗？**

答案：不会。clear 只删缓存引用，活跃请求仍保证页的生命周期。

**题 4：相同第 2 块 Token，不同第 1 块，可以共享第 2 块吗？**

答案：当前多层 causal 模型一般不可以；第二块 KV 的上下文依赖不同。完整前缀 key 防止这种误复用。

**题 5：ensure_capacity 返回 false，可否断言缓存一条都没被删？**

答案：不可以。可能已经淘汰 cache-only 页，但总量仍不足；请求没有部分新增页与全局没有副作用是不同保证。

下一篇：[任务 09：只计算真正用于采样的 logits 行](task_09_sample_rows_zh.md)。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

前置知识：[KV 与分页](from_pytorch/03_pages_and_packed.md)、[缓存引用语义的版本对照](from_pytorch/05_read_nanovllm_and_vllm.md#7-必须知道的实现差异)。
先弄清同请求 KV 复用，再学习跨请求前缀共享。

任务 09 起调试 logits 只返回采样行；本测试第二请求的映射为 `[1]`，返回一行词表。
三组性能对照已补在 [任务 10](task_10_prefix_benchmark_zh.md)。

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
| 页表变成 GPU 地址 | [`prepare_packed_model_input`](../mini_vllm/model_input.hpp#L41-L110) | [`paged_attention_decode`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1179-L1189) |
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
[`paged_attention_decode`](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1179-L1189)：

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
