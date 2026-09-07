# 开发任务 08：完整 Block Prefix Cache

## 1. Prefix Cache 解决什么问题

当多个请求拥有相同系统提示词或公共文档前缀时，前缀 Token 的 K/V 完全相同。普通
PagedAttention 虽然能分页存储 K/V，但每个请求仍会重新执行这段 Prefill。Prefix Cache
让新请求直接引用已经计算完成的物理 KV Block。

```text
请求 A: [公共前缀 16 Token] [A 的后缀]
                  │
                  └─ 物理 KV Block 3
                              ▲
请求 B: [公共前缀 16 Token] [B 的后缀]
```

请求 B 的 `block_table[0]` 直接写入 3，并把 `num_computed_tokens` 前移 16。

## 2. 为什么 Key 包含完整历史前缀

K/V 不只由当前 Block 的 Token 决定，还受到之前所有 Token 的位置和上下文影响。因此
不能只用当前 16 个 Token 作为 Key。本项目教学版使用：

```text
Key(block_i) = tokens[0 : (i + 1) * block_size]
```

即用从 Prompt 开头到当前 Block 末尾的完整 Token 向量作为有序 Map Key。生产系统通常
使用滚动哈希或内容哈希，降低 Key 的存储和比较成本。

## 3. 引用计数

每个缓存 Block 的引用由两部分组成：

- Prefix Cache 自身持有 1 个引用；
- 每个正在使用它的 Sequence 再持有 1 个引用。

请求结束只释放 Sequence 引用。只有 Cache 引用的 Block 可以被 LRU 驱逐；仍被活跃
请求使用的 Block 不会进入空闲队列。

## 4. 调度流程

Scheduler 第一次尝试接纳 Waiting 请求时：

1. 从逻辑 Block 0 开始连续查找缓存；
2. 每命中一页就增加引用计数并追加物理 Block ID；
3. 把 `num_computed_tokens` 前移命中的完整 Block 数；
4. 只为剩余 Token 分配新 Block 并进入 ModelRunner。

ModelRunner 完成 Token 后，Scheduler 将新完成的完整 Prompt Block 注册到 Cache。

当前实现始终留下至少一个 Token 重新计算，并且不缓存 Prompt 的最后一个 Block：

```cpp
cacheable_blocks = (num_prompt_tokens - 1) / block_size;
```

这样新请求不会向共享 Block 写入新的 K/V，也能重新得到最后位置的 logits。它牺牲部分
命中长度，换来无需 Copy-on-Write 的清晰正确性边界。

## 5. LRU 驱逐

当新请求缺少空闲 Block 时，BlockManager 查找 `ref_count == 1` 的最老 Cache Entry：

1. 删除 Cache Key；
2. 减少 Cache 引用；
3. 引用变成 0 后放回 Free List；
4. 重复直到容量足够或没有可驱逐 Block。

仍被请求共享的 Block 至少有 2 个引用，不会被驱逐。容量不足时，Sequence 的新增分配
仍保持全有或全无。

## 6. 测试结果

控制面测试使用 Block Size 4：

- 第一个请求产生两个可缓存 Block；
- 第二个请求命中两个 Block，只调度最后 1 个 Token；
- 第三个 13 Token 请求制造容量压力，Cache 按 LRU 驱逐两页；
- `clear_prefix_cache()` 后所有 Block 回到 Free List。

GPU 模型级测试使用 FP16、Page Size 16：

```text
second_prompt_tokens=18
prefix_cache_hit_blocks=1
second_scheduled_tokens=2
max_abs_logit_error=0.104279
generated_token_equal_cpu=true
```

这条测试证明共享的不只是 Block ID：第一页中原请求写入的 12 层 K/V 被第二个请求的
PagedAttention 直接读取，并与 CPU 完整前缀结果对齐。

## 7. 复现命令

```bash
make test_minivllm_control_plane
./test_minivllm_control_plane

make GPU_COMPUTE_CAPABILITY=86 test_gpt2_cuda_prefix_cache
CUDA_VISIBLE_DEVICES=0 ./test_gpt2_cuda_prefix_cache
```

## 8. 当前限制

- Key 使用完整 Token 向量，长 Prompt 下不如哈希高效；
- 只复用连续的完整前缀 Block；
- 最后一个 Prompt Block 总会重新计算；
- 没有 Partial Block Cache 和 Copy-on-Write；
- Cache 只存在于单个 Engine 生命周期内；
- 没有跨模型、跨进程或分布式 Cache。

这些限制是有意保留的。第一版只需要掌握内容寻址、引用计数、调度跳过、共享物理页和
LRU 驱逐五个核心概念。
