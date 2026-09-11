# 第 8 篇：从测试反推推理引擎的正确性要求

[返回学习目录](README.md) · 前置：[请求执行过程](06_request_walkthrough_zh.md)、[PyTorch/CUDA 对照](07_pytorch_cuda_bridge_zh.md)

本篇教你逐条解释仓库已有测试：输入为什么这样构造，断言保护哪项约定，怎样改错代码才能让它失败，测试通过以后仍有哪些结论不能下。

面试回答“做了单元测试”还不够。你需要能说出一个具体错误、一个能区分该错误的输入、一条对应断言，以及这条断言覆盖的范围。

## 1. 先用一个固定格式读测试

每个测试都写下以下六项：

```text
前提：测试开始时哪些状态必须成立？
输入：为什么选这些长度、页号、请求组合？
动作：调用了哪个真实函数？
预期：为什么正确结果必须是这个值？
反例：哪种具体错误会违反这个预期？
边界：通过后还不能说明什么？
```

“我看到了 assert”不等于理解了预期的来源。先手算，再读断言，最后运行。测试名字可以作为线索，结论必须以实际构造和检查为准。

## 2. 按正确性问题找到对应测试

| 要回答的问题 | 主要入口 | 实际执行范围 |
| --- | --- | --- |
| 页能分配、归还、复用吗 | [控制面测试](../../dev/test_mini_vllm_control_plane.cpp) | CPU，使用教学 ID |
| 分块 Prefill 和连续接纳正确吗 | 同一控制面测试 | CPU 调度与记账 |
| CPU 模型/Engine 能与 dense 对齐吗 | [CPU Engine 测试](../../dev/test_gpt2_engine.cpp) | checkpoint + CPU 模型 |
| 分页 Attention 地址和数值正确吗 | [CUDA Attention 测试](../../dev/cuda/test_paged_attention.cu) | GPU 单算子 + 独立 dense 参考 |
| 完整 CUDA Runner 正确吗 | [CUDA Runner 测试](../../dev/cuda/test_gpt2_cuda_model_runner.cu) | GPU 模型 + CPU dense |
| 采样行裁剪和 Graph 重放正确吗 | [sample rows 测试](../../dev/cuda/test_gpt2_cuda_sample_rows.cu) | 裁剪 Graph 与完整 Eager 对照 |
| Prefix 确实复用了正确内容吗 | [Prefix 测试](../../dev/cuda/test_gpt2_cuda_prefix_cache.cu) | 减少输入行 + CPU 输出对照 |
| PD 迁移后还能继续正确生成吗 | [PD 测试](../../dev/cuda/test_gpt2_pd_engine.cu) | 双 GPU 迁移 + 后续 Decode |
| 错误是否会被已有断言发现 | [本篇错误注入实验](examples/mutation_demo.py) | 临时 CPU 源码副本 |

本轮重新运行了控制面测试、错误注入实验、前两篇新增的 CPU/PyTorch/CUDA 微型模型实验。下面解释的完整 124M 模型 GPU 回归使用仓库已有源码与 [实测记录](../../benchmark/results/task09_11/validation.txt)，本轮没有重新跑完这套回归矩阵。

## 3. 第一个断言：跨页到底跨了什么

源码/记录：[dev/test_mini_vllm_control_plane.cpp，第 19—30 行](../../dev/test_mini_vllm_control_plane.cpp#L19)。

```cpp
static void test_block_allocation_release_and_reuse() {
    BlockManager manager(6, 4);
    auto a = request(1, 5, 1);
    auto b = request(2, 4, 1);
    assert(manager.ensure_capacity(*a, 5));
    assert(manager.ensure_capacity(*b, 4));
    assert(a->block_table().size() == 2);
    assert(manager.block_id_for_token(*a, 4) == a->block_table()[1]);
    assert(manager.slot_for_token(4) == 0);
    const std::vector<int> released = a->block_table();
    manager.release(*a);
    assert(manager.num_free_blocks() == 5);
```

前提：6 个物理页，每页 4 Token。A 申请 5 个 Token 容量，B 申请 4 个。

手算：A 需要 `ceil(5/4)=2` 页；B 需要 1 页。A 的 token index=4 位于逻辑页 1、页内 offset=0。因此 `block_id_for_token(a,4)` 必须等于 `a.block_table()[1]`。

这两条断言分别保护“选页”和“页内偏移”，不能只检查总页数。即使分配了两页，如果寻址仍读取第一页，页面数量也会看起来正确。

`release(a)` 后只剩 B 占一页，free 应为 `6−1=5`。它检查容量是否归还，仍不涉及 GPU 上 K/V 的内容。

**面试短答：** 我在页边界使用位置 4、页大小 4，分别检查逻辑页选择和页内偏移；再通过分配、释放后的空闲容量检查所有权记账。

## 4. 复用测试为什么不只断言“刚释放的页马上回来”

源码/记录：[dev/test_mini_vllm_control_plane.cpp，第 32—44 行](../../dev/test_mini_vllm_control_plane.cpp#L32)。

```cpp
    // Consume every free block. FIFO allocation need not reuse the most recently
    // released block first, but all released capacity must eventually be reusable.
    auto c = request(3, 20, 1);
    assert(manager.ensure_capacity(*c, 20));
    assert(c->block_table().size() == 5);
    assert(c->block_table()[3] == released[1]);
    assert(c->block_table()[4] == released[0]);
    manager.validate();

    bool rejected_double_free = false;
    try { manager.release(*a); }
    catch (const std::logic_error&) { rejected_double_free = true; }
    assert(rejected_double_free);
```

分配器使用 FIFO 空闲队列，而释放操作把页按逆序放回队尾。释放 A 之前，还可能有其他从未使用的空闲页排在前面。

因此测试让 C 申请 20 个 Token，也就是 5 页，取走全部空闲页，再检查 A 的两个旧页确实出现在结果中。这个输入规避了错误假设“最近释放页一定最先再次分配”。

双重释放测试捕获 `logic_error`，证明该场景下第二次 release 被拒绝。它不证明所有任意损坏页表都可事务式恢复。`BlockManager` 面向受控引擎内部调用，正常分配保证页表结构；测试结论不应扩成对任意外部内存破坏的容错承诺。

**与你的主线连接：** 第 6 篇回收后探针拿到 `[1,0]`，是当前 FIFO 和逆序归还共同产生的具体结果，不是所有页分配器必须遵守的统一规范。

## 5. 分块 Prefill：为什么选择 Prompt9、预算4

源码/记录：[dev/test_mini_vllm_control_plane.cpp，第 47—67 行](../../dev/test_mini_vllm_control_plane.cpp#L47)。

```cpp
static void test_chunked_prefill() {
    BlockManager manager(8, 4);
    Scheduler scheduler({1, 4}, manager);
    auto a = request(10, 9, 1);
    scheduler.add(a);

    auto first = scheduler.schedule();
    assert(first.items.size() == 1 && first.items[0].num_scheduled_tokens == 4);
    assert(first.items[0].phase == ExecutionPhase::Prefill);
    scheduler.commit(first, {-1});

    auto second = scheduler.schedule();
    assert(second.items[0].num_scheduled_tokens == 4);
    scheduler.commit(second, {-1});

    auto third = scheduler.schedule();
    assert(third.items[0].num_scheduled_tokens == 1);
    scheduler.commit(third, {777});
    assert(a->is_finished());
    assert(scheduler.is_finished());
    assert(manager.num_free_blocks() == manager.num_blocks());
```

9 除以 4 余 1，自然产生 `4、4、1` 三段。它比“Prompt4、预算4”多覆盖两个关键状态：

1. 前两轮模型输入尚未完成，返回标记必须为 -1。
2. 第三轮处理最后一行，才追加唯一输出并完成请求。

按 Sequence 规则推导：

| 时刻 | computed | total | 本轮标记 | 含义 |
| --- | --- | --- | --- | --- |
| 初始 | 0 | 9 | — | Prompt 尚未处理 |
| 第 1 轮提交后 | 4 | 9 | -1 | 部分 Prefill |
| 第 2 轮提交后 | 8 | 9 | -1 | 部分 Prefill |
| 第 3 轮标记输入完成 | 9 | 9 | 777 | 可以追加首输出 |
| 第 3 轮最终状态 | 9 | 10 | 已追加 | max_new_tokens=1，Finished |

这个测试的 777 是教学 ID，没有调用模型，也没有检查词表范围。不能把它的通过表述为“真实模型生成结果正确”。

代码直接断言 count、完成状态和空闲页恢复；中间 computed 表可由实现推导，但原测试没有逐行显式断言这张表中的每个数字。新请求执行实验补充了四轮 computed 的显式检查。

## 6. 读到 OOM 测试时，怎样避免夸大“原子性”

源码/记录：[dev/test_mini_vllm_control_plane.cpp，第 70—87 行](../../dev/test_mini_vllm_control_plane.cpp#L70)。

```cpp
static void test_oom_is_atomic_and_eos_releases_blocks() {
    BlockManager manager(2, 4);
    auto owner = request(30, 8, 1);
    auto blocked = request(31, 1, 1);
    assert(manager.ensure_capacity(*owner, 8));
    assert(!manager.ensure_capacity(*blocked, 1));
    assert(blocked->block_table().empty());
    assert(manager.num_free_blocks() == 0);
    manager.release(*owner);

    Scheduler scheduler({1, 4}, manager);
    auto eos_request = std::make_shared<Sequence>(
        32, std::vector<int>{1, 2}, SamplingParams{10, 42, false});
    scheduler.add(eos_request);
    auto output = scheduler.schedule();
    scheduler.commit(output, {42});
    assert(eos_request->is_finished());
    assert(manager.num_free_blocks() == manager.num_blocks());
```

两页容量先被 owner 用完，再给 blocked 请求申请 1 个 Token。预期：申请返回 false，blocked 页表仍为空，free 仍为 0。

这证明**关闭 Prefix Cache 的这个场景里，失败申请没有给目标请求留下半份页面分配**。随后释放 owner，EOS 请求生成配置中的 ID 42，立即完成并归还容量。

不要从测试函数名推导“任何 OOM 都完全没有副作用”。开启 Prefix Cache 时，`ensure_capacity` 会先尝试淘汰可驱逐缓存；即便最终仍不足，缓存集合也可能已经改变。这里没有构造和检查那个分支。

同样，`Scheduler::commit` 不是通用事务接口。它先执行 `mark_computed`，再检查部分 Prefill 的标记是否为 -1。传入非法采样标记触发异常时，计数可能已推进。当前正常调用链通过 Runner 协议提供合法结果，不应声称 commit 对任意非法输入自动回滚。

**面试更准确的表达：** 测试覆盖了分配不足时目标请求不留下部分页表，以及 EOS 后资源回收；对缓存淘汰副作用和异常恢复，需要分别定义和测试。

## 7. 连续批处理测试在保护什么调度行为

源码/记录：[dev/test_mini_vllm_control_plane.cpp，第 90—117 行](../../dev/test_mini_vllm_control_plane.cpp#L90)。

```cpp
static void test_continuous_admission_and_retirement() {
    BlockManager manager(12, 4);
    Scheduler scheduler({2, 5}, manager);
    auto a = request(20, 5, 2);
    auto b = request(21, 2, 1);
    auto c = request(22, 3, 1);
    scheduler.add(a);
    scheduler.add(b);
    scheduler.add(c);

    auto step1 = scheduler.schedule();
    assert(step1.items.size() == 1 && step1.num_batched_tokens == 5);
    scheduler.commit(step1, {900});
    assert(scheduler.num_running() == 1 && scheduler.num_waiting() == 2);

    auto step2 = scheduler.schedule();
    assert(step2.items.size() == 2);
    assert(step2.items[0].sequence == a && step2.items[0].phase == ExecutionPhase::Decode);
    assert(step2.items[1].sequence == b && step2.items[1].phase == ExecutionPhase::Prefill);
    scheduler.commit(step2, {901, 902});
    assert(a->is_finished() && b->is_finished());
    assert(scheduler.num_running() == 0 && scheduler.num_waiting() == 1);

    auto step3 = scheduler.schedule();
    assert(step3.items.size() == 1 && step3.items[0].sequence == c);
    scheduler.commit(step3, {903});
    assert(c->is_finished() && scheduler.is_finished());
    assert(manager.num_free_blocks() == manager.num_blocks());
```

A Prompt5、生成2；B Prompt2、生成1；C Prompt3、生成1。预算5、最多两条活跃请求。

第 1 轮 A 用满预算。第 2 轮 A 只需 Decode1，留下4个额度，B 可以进入并完成 Prefill2。两者都达到输出限制，C 在下一轮进入。

关键断言有两种：

| 断言 | 保护什么 | 可能暴露什么错误 |
| --- | --- | --- |
| 第 2 轮第一个 item 是 A/Decode | 已有活跃请求先推进 | 新请求持续插队导致 Decode 等待 |
| 第 2 轮第二个 item 是 B/Prefill | 利用剩余额度接纳新请求 | 固定 Batch 必须整批完成才接纳 |
| 完成后 running=0,waiting=1 | 退场与等待队列更新 | 完成请求未移除或等待项被误删 |
| 第 3 轮只有 C | 后续请求继续前进 | 资源/队列泄漏造成无进展 |

这是调度策略的一个具体轨迹，不证明所有负载上的公平性、吞吐或 P99 延迟。那些需要独立负载与计时实验。

## 8. Prefix CPU 测试：命中、引用与淘汰分别看哪里

第一条 Prompt 是 `[1,2,3,4,5,6,7,8,90]`，第二条只把末 Token 改为 91。页大小 4，前两页完整且相同，最后一页保留计算。

源码/记录：[dev/test_mini_vllm_control_plane.cpp，第 139—149 行](../../dev/test_mini_vllm_control_plane.cpp#L139)。

```cpp
        SamplingParams{1, -1, false});
    scheduler.add(second);
    auto second_step = scheduler.schedule();
    assert(second->num_computed_tokens() == 8);
    assert(second_step.items[0].num_scheduled_tokens == 1);
    assert(second_step.items[0].phase == ExecutionPhase::Prefill);
    assert(second->block_table()[0] == cached_block_ids[0]);
    assert(second->block_table()[1] == cached_block_ids[1]);
    assert(manager.prefix_cache_hit_blocks() == 2);
    scheduler.commit(second_step, {901});
    assert(second->is_finished());
```

`computed=8` 说明两个完整页对应的输入已被复用；`count=1` 说明本轮只处理剩余末 Token；页号相同说明共享映射发生；hit_blocks=2 是缓存命中计数。

这四类证据互相补充。只检查 hit counter，不能证明模型真的少算了输入；只检查 count，也不能证明复用的是正确页。

随后 pressure 请求不共享前缀，13 Token 需要四页。当前只有两页空闲，需要淘汰两个仅缓存持有的页。测试检查旧缓存清空、容量被用于当前请求，完成后新完整前缀又被保留。

这里能够证明“这个压力场景可淘汰缓存并继续工作”，但由于两个旧缓存页最终都必须淘汰，**不能仅凭该结果证明多个候选中一定选择了最老者**。若要专门验证 LRU 选择次序，应构造只需淘汰一个页、且候选 last_used 不同的输入。该点是测试设计练习，本轮未增加引擎功能。

## 9. 单算子测试：为什么故意打乱物理页

源码/记录：[dev/cuda/test_paged_attention.cu，第 132—151 行](../../dev/cuda/test_paged_attention.cu#L132)。

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

三个请求各自页表都不连续，例如第一条 `[11,2,7,0]`。长度包含 15/16/17、31/32/33，覆盖页边界前、边界末和下一页首。

为什么同时需要这两类输入？

- 只有边界长度、页号恰好连续：把页表读取错误写成连续地址，可能仍算对。
- 只有乱序页号、上下文永远不跨页：第二页索引错误可能永远不执行。
- 两者结合：页内偏移、逻辑页选择、实际物理页映射都有机会影响结果。

测试只选 layer_index=1、num_layers=2，也让地址计算必须经过 layer 维。不能据此说所有层索引和所有 head size 都已穷举。

## 10. 如何确保 KV 写入 kernel 真的被测试到

构造 dense K/V 后，测试先把对应历史内容散布到分页池；遇到当前最后位置时，保留 new_k/new_v，并把缓存中的该位置清零。

源码/记录：[dev/cuda/test_paged_attention.cu，第 224—235 行](../../dev/cuda/test_paged_attention.cu#L224)。

```cpp
                            if (token == context_lengths[request] - 1) {
                                const std::size_t current =
                                    (static_cast<std::size_t>(request) *
                                         num_heads +
                                     head) *
                                        head_size +
                                    dimension;
                                new_key[current] = dense_keys[dense];
                                new_value[current] = dense_values[dense];
                                key_cache[cache] = 0.0f;
                                value_cache[cache] = 0.0f;
                            }
```

若预先把“当前 Token 的正确 KV”也完整放进池里，即使写入 kernel 被错误删除，Attention 仍可能读到正确值。将当前槽清零会破坏这种掩盖，让实际写入成为得到正确结果的必要条件。

执行后测试不仅复制 Attention 输出，也把 K/V 池读回，检查当前写入位置。最后的通过条件是：

源码/记录：[dev/cuda/test_paged_attention.cu，第 314—324 行](../../dev/cuda/test_paged_attention.cu#L314)。

```cpp
        std::cout
            << "CUDA PagedAttention correctness passed: "
            << "lengths=1,7,15,16,17,31,32,33,64 "
            << "max_abs_error=" << global_max_abs_error
            << " max_rel_error=" << global_max_rel_error
            << " max_kv_write_error=" << global_max_kv_write_error
            << '\n';
        return global_max_abs_error < 2e-4 &&
                       global_max_kv_write_error == 0.0
                   ? 0
                   : 1;
```

因此运行测试时要看退出码。该程序的输出文字含 `correctness passed`，但真正成功与否由后面的误差判断决定，不能只 grep 这一句字符串。

原测试检查了当前槽的写入值；它没有逐元素检查所有非目标槽都保持原状。第 7 篇 Python 实验演示了这种“写入区域之外不变”的检查方法，但它是 CPU 参考检查，不等于已给 CUDA 原测试补上这项覆盖。

## 11. Dense 参考怎样避免与分页实现一起犯错

原单算子参考 [dense_reference](../../dev/cuda/test_paged_attention.cu#L64) 按 `[request,token,head,dimension]` 遍历连续逻辑 K/V，使用 double 计算点积、Softmax 和 V 加权和。

真正被比较的是：同一 query 和逻辑历史，在 dense 存储与分页存储下得到的输出是否一致。

需要保持判断分寸：测试准备分页池时也有一份 `cache_offset` 展开函数，若准备端与 kernel 完全抄到同一个布局错误，部分错误仍可能相互抵消。因此第 7 篇还给出具体五维下标、手算元素 711、原始 flatten 检查，帮助审查布局假设。

**测试设计原则：** 参考越复用被测实现的关键逻辑，越容易共享错误。不是绝对禁止复用辅助代码，而是要清楚当前比较独立在哪一层。

## 12. 完整模型测试：如何从采样行找正确参考位置

源码/记录：[dev/cuda/test_gpt2_cuda_model_runner.cu，第 72—98 行](../../dev/cuda/test_gpt2_cuda_model_runner.cu#L72)。

```cpp
    for (std::size_t logit_row = 0; logit_row < logit_rows.size(); ++logit_row) {
        const std::size_t row = logit_rows[logit_row];
        const std::size_t item_index =
            final_input.scheduled_item_indices[row];
        const Sequence& sequence = *output.items[item_index].sequence;
        const int prefix_length = final_input.positions[row] + 1;
        gpt2_forward_dense_with_workspace(
            &model, sequence.token_ids().data(), 1, prefix_length,
            &reference_workspace);
        const float* expected =
            reference_workspace.acts().logits +
            static_cast<std::size_t>(prefix_length - 1) *
                model.config.padded_vocab_size;
        const float* actual =
            gpu_logits.data() + logit_row * model.config.padded_vocab_size;
        for (int token_id = 0; token_id < model.config.vocab_size;
             ++token_id) {
            validation.max_abs_logit_error = std::max(
                validation.max_abs_logit_error,
                std::abs(static_cast<double>(actual[token_id]) -
                         expected[token_id]));
        }
        if (argmax(actual, model.config.vocab_size) !=
            argmax(expected, model.config.vocab_size)) {
            ++validation.argmax_mismatches;
        }
    }
```

这是本项目最值得精读的验证链之一：

```text
logit_row：压缩后第几行 logits
  → logit_rows[logit_row]：原 Packed 行
  → scheduled_item_indices[row]：属于哪个 item
  → output.items[item_index].sequence：属于哪个请求
  → positions[row]+1：该 query 的完整逻辑前缀长度
  → CPU dense 前向最后位置：参考 logits
```

以第 6 篇第 3 轮 B 为例：压缩行1 → Packed行4 → item1 → B → prefix_length19。参考必须运行 B 的前19个 Token，而不是整批五行，也不是 B 的 Packed 行号4对应的前5个 Token。

`expected` 与 `actual` 都使用 padded vocab size 作为行跨度，但循环只比较 vocab size 个有效列。行跨度和语义列数是两个参数，不能替换成同一个值。

该验证函数在 `scheduler.commit` 之前比较，此时 Sequence 还没有追加新输出；因此可直接用原输入前缀重算参考。若把验证移到 commit 后，需要重新检查哪部分 Token 属于本轮输入。

## 13. 为什么既比较 logits，又比较 Argmax

模型测试比较有效词表上最大绝对误差，也统计 Argmax 不一致次数。两者回答不同问题：

| 检查 | 能发现什么 | 单独使用的不足 |
| --- | --- | --- |
| 有效词表 logits 误差 | 数值结果偏离参考 | 小误差也可能改变非常接近的最高分次序 |
| Argmax / Greedy ID | 本轮实际生成决策不同 | 大量非最大值错误也可能不改变结果 |
| 多步输出序列 | 错误如何沿生成循环传播 | 首次不同以后难以定位最早内部偏差 |
| 内存检查 | 被执行路径上的非法设备访问 | 合法地址中的错误数据仍可能通过 |

第 7 篇的 `pruned+100` 反例不改变 Argmax，但 logits 明显不同。它说明 Greedy 相同不足以证明数值相同；不表示所有 logits 变化都同样影响生成语义。

原 Runner 测试的结束条件：

源码/记录：[dev/cuda/test_gpt2_cuda_model_runner.cu，第 230—240 行](../../dev/cuda/test_gpt2_cuda_model_runner.cu#L230)。

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
```

FP32 的阈值 0.2、其他类型阈值 3.0，是**当前测试选择的绝对误差门限**，不是浮点格式理论保证。BF16 分支没有强制 Argmax mismatches=0；看到总的 PASS 不能据此说 BF16 的所有 Greedy 输出都与 FP32 相同。

已有实测的误差可能远小于门限，面试时要区分“允许通过的阈值”和“某次测到的最大误差”。固定输入上的通过也不覆盖所有模型、长度、随机种子。

## 14. 采样行测试为何固定 N、改变 R

源码/记录：[dev/cuda/test_gpt2_cuda_sample_rows.cu，第 18—24 行](../../dev/cuda/test_gpt2_cuda_sample_rows.cu#L18)。

```cpp
        BlockManager pruned_blocks(8, 16), full_blocks(8, 16);
        GPT2CudaModelRunner pruned(config, model.params_memory, model.num_parameters, pruned_blocks, 2, 4, 16);
        config.enable_sample_row_pruning = false;
        config.enable_cuda_graph = false;
        GPT2CudaModelRunner full(config, model.params_memory, model.num_parameters, full_blocks, 2, 4, 16);
        const std::vector<std::vector<int>> lengths{{8}, {4}, {2, 2}, {2, 8}, {8}};
        const std::vector<std::vector<int>> expected_rows{{}, {3}, {1, 3}, {1}, {}};
```

测试在两个 Runner 间比较：裁剪开启且 Graph 开启；完整行计算且 Graph 关闭。它同时覆盖组合路径的行为，但不能只凭这一个比较把差异归因到裁剪或 Graph 中某一个开关。

五组输入都让 N=4，预期如下：

| 组 | 请求 Prompt 长度 | 各请求本轮输入数 | 完成输入的请求 | sample rows | R |
| --- | --- | --- | --- | --- | --- |
| 0 | `[8]` | `[4]` | 无 | `[]` | 0 |
| 1 | `[4]` | `[4]` | 第一个 | `[3]` | 1 |
| 2 | `[2,2]` | `[2,2]` | 两个 | `[1,3]` | 2 |
| 3 | `[2,8]` | `[2,2]` | 第一个 | `[1]` | 1 |
| 4 | `[8]` | `[4]` | 无 | `[]` | 0 |

第 1、3 组都使用 `(N=4,R=1)`，但采样行从3变为1、Token 输入也改变。如果重放时忘记刷新行索引，同形状图可以成功启动，却会读取错误 hidden 行。

源码/记录：[dev/cuda/test_gpt2_cuda_sample_rows.cu，第 41—58 行](../../dev/cuda/test_gpt2_cuda_sample_rows.cu#L41)。

```cpp
            const auto samples_a = pruned.run(a), samples_b = full.run(b);
            assert(samples_a == samples_b);
            assert(pruned.last_logit_token_indices() == expected_rows[test]);
            assert(pruned.num_cuda_graphs() == std::min(test + 1, std::size_t{3}));
            const auto la = pruned.last_logits_for_testing(), lb = full.last_logits_for_testing();
            assert(la.size() == expected_rows[test].size() * model.config.padded_vocab_size);
            assert(lb.size() == 4ul * model.config.padded_vocab_size);
            for (std::size_t row = 0; row < expected_rows[test].size(); ++row) {
                for (int v = 0; v < model.config.vocab_size; ++v) {
                    const double error = std::abs(double(la[row * model.config.padded_vocab_size + v]) -
                        lb[expected_rows[test][row] * model.config.padded_vocab_size + v]);
                    max_error = std::max(max_error, error);
                }
            }
            for (auto& item : a.items) pruned_blocks.release(*item.sequence);
            for (auto& item : b.items) full_blocks.release(*item.sequence);
        }
        assert(max_error < 0.005);
```

逐条解释：

- `samples_a==samples_b`：返回给 Scheduler 的 item 结果相同，包括 -1 标记。
- `last_logit_token_indices==expected_rows`：CPU 算出的行映射符合手算。
- `num_cuda_graphs==min(test+1,3)`：前三种 R 各建立图，后续复用已有形状，缓存不继续增加。
- logits 长度：确认裁剪后的缓冲有效行数确实是 R，而完整基线是4。
- 按 `expected_rows[row]` 比较有效词表：确认每个压缩输出对应正确原始行。

R=0 时 logits 为空，这一组的 logits 循环不会检查主体 KV 的数值。它覆盖零输出分支、图缓存和标记返回；要证明这次部分 Prefill 建立的 KV 正确，还需要后续生成或其他完整路径对照。

## 15. Prefix GPU 测试为何同时检查“少算”和“算对”

先运行首请求留下一个完整16 Token前缀页，再用相同前16 Token加两个不同后缀 Token构造第二个 Prompt18。

源码/记录：[dev/cuda/test_gpt2_cuda_prefix_cache.cu，第 78—86 行](../../dev/cuda/test_gpt2_cuda_prefix_cache.cu#L78)。

```cpp
    auto second = engine.add_request(
        2, second_prompt, SamplingParams{1, -1, false});
    const CudaEngineStepResult second_step = engine.step();
    assert(second_step.num_batched_tokens == 2);
    assert(second_step.scheduled_token_counts.size() == 1);
    assert(second_step.scheduled_token_counts[0] == 2);
    assert(second->is_finished());
    assert(second->token_ids().back() == expected_second);
    assert(engine.prefix_cache_hit_blocks() == 1);
```

这些断言说明第二条请求本轮输入从18减少到2，确实命中一个页，并且首输出与 CPU dense 参考一致。随后继续比较完整有效词表 logits：

源码/记录：[dev/cuda/test_gpt2_cuda_prefix_cache.cu，第 88—104 行](../../dev/cuda/test_gpt2_cuda_prefix_cache.cu#L88)。

```cpp
    const std::vector<float> gpu_logits =
        engine.model_runner().last_logits_for_testing();
    gpt2_forward_dense_with_workspace(
        &model, second_prompt.data(), 1,
        static_cast<int>(second_prompt.size()), &reference);
    const float* expected_logits = reference.acts().logits +
        (second_prompt.size() - 1) * model.config.padded_vocab_size;
    assert(engine.model_runner().last_logit_token_indices() == std::vector<int>{1});
    const float* actual_logits = gpu_logits.data();
    double max_abs_error = 0.0;
    for (int token = 0; token < model.config.vocab_size; ++token) {
        max_abs_error = std::max(
            max_abs_error,
            std::abs(static_cast<double>(actual_logits[token]) -
                     expected_logits[token]));
    }
    assert(max_abs_error < 0.2);
```

`sample_rows={1}` 来自“本轮仅有两行，最后一行为索引1”。如果误以为它还在原 Prompt 的第17行，就会从错误缓冲行取数据。

这个测试证明构造输入上的复用正确与工作量下降。它没有测 TTFT；性能效果应看单独的 Prefix benchmark，不能把“少算16行”直接当作“延迟减少16/18”。

## 16. PD：怎样让“复制到了错误页”无法蒙混过关

源码/记录：[dev/cuda/test_gpt2_pd_engine.cu，第 33—45 行](../../dev/cuda/test_gpt2_pd_engine.cu#L33)。

```cpp
    auto source = std::make_shared<Sequence>(1, prompt(17), SamplingParams{4, -1, true});
    auto target = std::make_shared<Sequence>(1, prompt(17), SamplingParams{4, -1, true});
    Sequence blocker(99, prompt(1), {1, -1, true});
    assert(src_blocks.ensure_capacity(*source, 20));
    assert(dst_blocks.ensure_capacity(blocker, 16));
    assert(dst_blocks.ensure_capacity(*target, 20));
    assert(source->block_table()[0] != target->block_table()[0]);
    SchedulerOutput prefill{{{source, ExecutionPhase::Prefill, 17}}, 17};
    const auto first = src.run(prefill)[0]; source->mark_computed(17); source->append_token(first);
    const auto stats = src.copy_kv_to(dst, *source, *target, 17);
    assert(stats.num_pages == 2 && stats.payload_bytes == 2 * 2 * 16 * 12 * 768 * 2);
    assert(src_blocks.num_free_blocks() == 4); // copy 不负责回收源页。
    target->mark_computed(17); target->append_token(first);
```

目标端先分配一个 blocker，占住页0。这样源、目标请求的第一个物理页号不同，测试强制要求按逻辑页逐一对应源和目标地址。

P 端处理17 Token并得到首输出后，调用 copy；目标恢复 computed17并追加同一首输出。随后两端各执行三轮 Decode：

源码/记录：[dev/cuda/test_gpt2_pd_engine.cu，第 46—57 行](../../dev/cuda/test_gpt2_pd_engine.cu#L46)。

```cpp
    for (int i = 0; i < 3; ++i) {
        SchedulerOutput a{{{source, ExecutionPhase::Decode, 1}}, 1};
        SchedulerOutput b{{{target, ExecutionPhase::Decode, 1}}, 1};
        const int sa = src.run(a)[0], sb = dst.run(b)[0]; assert(sa == sb);
        const auto la = src.last_logits_for_testing(), lb = dst.last_logits_for_testing();
        assert(la.size() == lb.size());
        double error = 0;
        for (int v = 0; v < model.config.vocab_size; ++v)
            error = std::max(error, std::abs(double(la[v]) - lb[v]));
        assert(error < 1e-5);
        source->mark_computed(1); source->append_token(sa);
        target->mark_computed(1); target->append_token(sb);
```

每轮同时比较采样 ID 和有效词表 logits，避免“虽然 KV 复制错，但当前最大值恰好相同”漏检。只比较迁移字节数不能检查内容与逻辑页顺序。

`payload_bytes` 的常量式断言依赖本测试的模型和存储配置：两页 × K/V两份 × 每页16 Token × 12层 × 768通道 × 2字节。它不是任何模型都成立的通用常量；通用公式需要代入实际层数、KV头数、head dimension和存储类型。本项目使用 MHA，因此 H×D=C。

`src_blocks.num_free_blocks()==4` 说明 copy 本身没有回收源页。把复制与释放分开，才能在目标准备好后由上层决定交接时机。

## 17. PD 流水测试的 concurrency 字段能证明什么

源码/记录：[dev/cuda/test_gpt2_pd_engine.cu，第 86—110 行](../../dev/cuda/test_gpt2_pd_engine.cu#L86)。

```cpp
    std::size_t transfers = 0, concurrent = 0, iterations = 0;
    while (!pd.is_finished()) {
        assert(++iterations < 100);
        auto step = pd.step();
        transfers += step.handed_off.size(); concurrent += step.concurrent_submissions;
        assert(step.sampled_request_ids.size() == step.sampled_token_ids.size());
    }
    while (!single.is_finished()) single.step();
    assert(transfers == lengths.size() - 1 && concurrent > 0);
    GPT2DenseInferenceWorkspace workspace(model.config, 1, 64);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = *requests[i];
        assert(request.stage == PDStage::Finished && request.sequence->is_finished());
        assert(request.sequence->block_table().empty());
        assert(completion(*request.sequence) == completion(*references[i]));
        auto tokens = prompt(lengths[i], i + 1);
        for (int expected : completion(*request.sequence)) {
            gpt2_forward_dense_with_workspace(&model, tokens.data(), 1, tokens.size(), &workspace);
            const float* logits = workspace.acts().logits + (tokens.size() - 1) * model.config.padded_vocab_size;
            assert(expected == std::max_element(logits, logits + model.config.vocab_size) - logits);
            tokens.push_back(expected);
        }
    }
    assert(requests[0]->transfer.payload_bytes == 0);
    assert(pd.prefill_free_blocks() == 4 && pd.decode_free_blocks() == d_blocks);
```

测试在两种 D 容量下运行，检查请求最终完成、页表清空、输出与单卡及 CPU Greedy 参考相同。长度1、16、17、31、32、33覆盖若干页边界；首请求只生成一个 Token，不需要迁移。

`concurrent_submissions>0` 表示执行过两端并发提交的调度分支。它不等于两个设备 kernel 在物理时间上必定有重叠，也不等于端到端更快。真实时间重叠要看 trace，速度要看同条件基准，参见 [性能分析手册](05_profiling_lab_zh.md)。

受限容量场景能在迭代上限内完成，说明该输入下背压与推进没有卡死；它不证明所有请求组合下的无饥饿性质。

## 18. 三个受控错误：亲自验证断言能否发现问题

脚本 [mutation_demo.py](examples/mutation_demo.py) 会把当前控制面头文件和原测试复制到 `/tmp`，先验证基线，然后每次只改错一处，编译并运行。项目源码不改动。

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda activate zyf1
python doc/interview/examples/mutation_demo.py
```

源码/记录：[doc/interview/examples/mutation_demo.py，第 9—19 行](examples/mutation_demo.py#L9)。

```python
MUTATIONS = [
    ('M1 页号除法误写成取模', 'block_manager.hpp',
     'const std::size_t logical_block = token_index / block_size_;',
     'const std::size_t logical_block = token_index % block_size_;'),
    ('M2 忽略本轮 Token 预算', 'scheduler.hpp',
     'const std::size_t count = std::min(sequence->pending_tokens(), budget);',
     'const std::size_t count = sequence->pending_tokens();'),
    ('M3 分配时多记一次引用', 'block_manager.hpp',
     'block.ref_count = 1;', 'block.ref_count = 2;'),
]

```

| 错误 | 改错后意味着什么 | 首个失败断言 | 为什么能检测 |
| --- | --- | --- | --- |
| M1：页号除法写成取模 | 位置4、页大小4错误选择逻辑页0 | `block_id_for_token(a,4)==table[1]` | 输入正好跨页，预期页与错误页不同 |
| M2：忽略 Token 预算 | Prompt9 第一轮直接安排9行 | 首轮 count 必须是4 | 预算小于 pending，错误会改变调度计划 |
| M3：初始引用记为2 | 请求释放一次后引用还剩1，页不回 free | 释放 A 后 free 必须是5 | 页面泄漏直接反映到容量 |

源码/记录：[doc/interview/examples/mutation_demo_output.txt，第 1—8 行](examples/mutation_demo_output.txt#L1)。

```text
基线 PASS: 未修改的控制面测试正常通过
M1 页号除法误写成取模: 被现有断言检测，returncode=-6
control_test: <临时目录>/dev/test.cpp:26: void test_block_allocation_release_and_reuse(): Assertion `manager.block_id_for_token(*a, 4) == a->block_table()[1]' failed.
M2 忽略本轮 Token 预算: 被现有断言检测，returncode=-6
control_test: <临时目录>/dev/test.cpp:54: void test_chunked_prefill(): Assertion `first.items.size() == 1 && first.items[0].num_scheduled_tokens == 4' failed.
M3 分配时多记一次引用: 被现有断言检测，returncode=-6
control_test: <临时目录>/dev/test.cpp:30: void test_block_allocation_release_and_reuse(): Assertion `manager.num_free_blocks() == 5' failed.
PASS: 三个错误各自编译成功、运行失败；临时副本已清理，项目源码未改动。
```

`returncode=-6` 在这里来自断言触发进程终止，是**错误被发现的预期结果**；整个教学脚本成功退出，表示基线通过且三个错误均被断言发现。脚本关闭 core dump，并清理临时副本。

脚本还要求每个错误版本先编译成功、stderr 含断言信息，避免把语法错误或找不到二进制误当成测试发现了逻辑错误。

这些是本轮人为构造的教学错误，不是三次真实线上事故。面试可以说“我通过受控错误检查了这些断言的区分能力”，不要改写成公司客户场景的故障经历。

## 19. 为什么不能关闭 assert 后再运行这些测试

仓库这些 C++ 测试大量使用 `assert`，其中部分断言表达式还包含必要动作，例如：

```cpp
assert(manager.ensure_capacity(*a, 5));
```

加 `-DNDEBUG` 不只是停止比较返回值，也会让整个表达式不执行，导致前提状态改变。所以学习命令明确不加该选项，Python 脚本也不要加 `-O` 关闭 Python assert。

运行控制面原测试的最小命令：

```bash
c++ -std=c++17 -O0 -g -gdwarf-4 \
  dev/test_mini_vllm_control_plane.cpp -o /tmp/zyf_deepening_control_tests
/tmp/zyf_deepening_control_tests
```

从工程审查角度，这种把副作用放入 assert 的写法值得注意；本篇任务是解释当前测试，本轮没有重构生产测试体系。

## 20. 怎样从失败找到第一个分歧

| 最先观察到的问题 | 优先检查 | 为什么 |
| --- | --- | --- |
| scheduled count 或 phase 错 | Scheduler 的 pending、budget、队列顺序 | 计划已经错误，暂不用进入 kernel |
| positions/slot 与手算不同 | ModelInput 的 computed+offset、页表 | 输入模型前的地址协议已错 |
| 元数据对，但当前 KV 写错 | write kernel 的 source/destination offset | 缩小到缓存写入或布局 |
| KV 正确，Attention 输出不同 | context、Q、scale、Softmax、读取索引 | 输入缓存与读取计算分开验证 |
| 所有 hidden 正确，logits 行错误 | sample rows、Gather、LM Head 参数 | 优先检查 N/R 映射与输出行跨度 |
| 模型结果正确，下一轮状态错 | commit、append、finished/release | 错误发生在结果消费阶段 |

这是排查顺序，不表示当前测试会自动导出所有中间值。需要观察 hidden 或每层 KV 时，应在最小可复现输入中添加定点检查，再定位最早偏离参考的一层。不要一上来打印整个模型所有 Tensor。

## 21. 尚未由现有测试充分证明的内容

下面这些限制应当能说清楚，读测试时也可以自己发现：

| 当前证据 | 不能直接推出的结论 | 进一步验证应怎样设计 |
| --- | --- | --- |
| FIFO 下固定页号复用通过 | 任意页表破坏可恢复 | 定义非法输入契约，再检查完整状态 |
| OOM 时 blocked 页表为空 | 开启缓存的失败调用完全无副作用 | 记录失败前后缓存、引用与目标状态 |
| LRU 压力下两页全部被淘汰 | 候选选择一定按最老顺序 | 只需淘汰一页，控制候选使用时间 |
| R=0 的空 logits 正确 | 那一轮所有 KV 数值一定正确 | 接续下一轮并与完整参考比较 |
| CPU Python 非目标槽保持不变 | CUDA 写入没有破坏其他槽 | 设备缓存前后区域比较或哨兵检查 |
| 某输入 FP16 Greedy 相同 | 所有输入、精度模式都相同 | 多种长度/数值场景，分别记录误差与决策 |
| 双端并发提交、功能正确 | PD 已经带来性能收益 | 真实时间线及同条件端到端基准 |

这些是当前证据的边界与可选学习题，不是本轮承诺新增的引擎功能。先把已验证路径讲扎实，再按具体改动选择需要补充的验证。

## 22. 面试回答模板：从“测了”到“为什么可信”

可以按四句话回答：

> 我先定义这项功能必须保持的约定，例如分页后同一逻辑前缀应产生与 dense 一致的有效词表 logits。然后构造能暴露错误的输入，例如打乱页表、跨越16 Token边界、在一个 Batch 混合 Decode 和 Prefill。验证时既看元数据与资源状态，也比较数值输出和实际生成决策，设备访问另用内存检查。最后说明覆盖条件和限制，避免把固定输入通过理解为所有配置都已验证。

再准备三个可以展开的例子：

1. **页管理：** M1 除法误写取模，位置4对应的页选择断言立即失败。
2. **采样行：** 固定 N=4、R=1，rows 从3变1，检查 Graph 重放时的动态内容刷新。
3. **PD：** 目标预占一页迫使源/目标页号不同，迁移后继续多轮 Decode 并比较 logits。

前一个是本轮实际运行的受控错误；后两个对应仓库已有 GPU 回归设计与记录。讲个人参与时按真实完成的工作表达。

## 23. 闭卷题与答案

**题 1：测试使用连续页表 `[0,1,2]`，能验证页表寻址吗？**

只能覆盖这一种映射，无法区分“正确读页表”和“错误假设物理连续”这两种实现。应使用乱序映射并跨越页边界。

**题 2：为什么把当前 KV 槽预先清零？**

让当前写入成为得到正确结果的必要步骤。若提前装入正确值，缺失写入 kernel 也可能通过输出对照。

**题 3：Graph 缓存数量没增加，足以说明重放正确吗？**

不够。它只能说明缓存行为符合预期；还要改变相同形状下的动态输入和行索引，验证数值与结果映射。

**题 4：比较 logits 的循环为什么只跑 V 列，但行跨度用 Vp？**

Vp 是实际矩阵存储宽度，决定下一行起点；V 是有效 Token 数，决定哪些列有词表语义。

**题 5：Prefix hit_blocks 增加且输出相同，能证明减少计算吗？**

还要看 scheduled token 数或实际执行工作量。错误实现可能记了命中但仍重算全部 Prompt；只看输出不会发现性能路径失效。

**题 6：mutant 编译失败，算测试发现了语义错误吗？**

不算。本篇要求变体先编译成功，再由运行时断言拒绝错误行为。编译错误只能说明该变体不是有效可执行程序。

**题 7：一个程序打印 PASS、退出码却是1，应该怎样记录？**

记录失败并检查最终条件。日志文本不是成功状态的唯一来源，尤其本项目单算子程序在输出误差后才计算返回值。

**题 8：测试中的“原子 OOM”意味着 commit 出错也会回滚吗？**

不意味着。它们是不同函数和契约；原测试只检查某个分配不足场景。commit 的实际语句顺序还可能在发现非法标记前推进计数。

## 24. 本篇结束时应该能独立完成什么

任选一个真实测试，先遮住断言，用输入手算结果，再解释每条断言为什么成立。然后提出一种**具体可编译的错误改动**，预测首个失败位置；最后写出该测试未覆盖的一条边界。

完成这一步，你就能把“我做了测试”展开成可核查的技术回答。继续复述时，回到 [项目面试手册](02_project_defense_zh.md) 的三分钟介绍，把一个测试例子接到对应功能后面即可。
