# 面试手册 02：项目介绍、追问与证据

[面试学习目录](README.md) · 上一篇：[固定版本 vLLM 对照](01_vllm_source_map_zh.md) · 下一篇：[代码补全实验](03_coding_labs_zh.md)

本篇提供可以反复复述的回答骨架。推荐每张卡按“结论 → 原因 → 项目实现 → 证据/边界”讲四句；先理解，再压缩成自己的表达。

口述稿描述的是仓库已经实现并验证的能力。你的个人负责范围、实际调试经历和上游贡献，应按真实参与情况表达；不要把这里的教学推演当成亲历事件。

## 1. 面试官在这个项目中会确认什么

| 层次 | 常见提问 | 你需要提供的材料 |
| --- | --- | --- |
| 项目真实性 | 模型、设备、入口是什么？ | 代码路径、运行命令、固定实验 |
| 基础理解 | KV 为什么正确，分页解决什么？ | 因果推导、地址公式、边界手算 |
| 工程能力 | 请求结束、页不足、共享冲突怎么办？ | 状态更新与资源所有权 |
| 优化能力 | 哪一步变快，怎么证明？ | 同配置 A/B、原始点、trace |
| 判断力 | 负优化如何解释？哪些没做？ | 测量范围与合理后续方向 |
| 可迁移性 | 与 vLLM 如何对应？ | 固定版本源码差异 |

背诵的目标不是每个函数都能默写，而是每个结论都能落到一条代码路径和一个验证方法。

## 2. 30 秒项目介绍

> 这是一个基于 llm.c 的教学型 Mini-vLLM C++/CUDA 推理引擎。我围绕 GPT-2 串起请求调度、分页 KV Cache、packed Prefill 和 GPU 执行，并加入采样行裁剪、CUDA Graph、Prefix Cache 和功能性双卡 PD。验证上同时使用 CPU 完整前缀参考、生成一致性和性能 A/B。项目重点是理解推理引擎的数据流和优化方法；当前短负载里 PD 更慢，这个结果也保留了完整证据。

**记忆骨架：** 来源与目标 → 核心链路 → 两项优化 → 如何验证 → 一个边界。

如果只能再补一句，补自己实际最熟悉的优化，而不要把所有功能名再报一遍。推荐从“采样行裁剪”或“Prefix Cache”选一个展开。

## 3. 三分钟项目介绍

> 我的框架开发背景更多在算子、执行链路和图编译。这个项目让我补上模型前向外面的服务状态：同一时刻哪些请求运行、历史 KV 放在哪里、请求完成后如何回收空间。
>
> 我在 llm.c GPT-2 基础上建立了 Sequence、BlockManager、Scheduler 和 ModelRunner。Sequence 分开保存已知 Token 和已计算位置；Scheduler 按 Token budget 选择本轮工作；BlockManager 用逻辑页表映射物理 KV 页；Runner 把计划变成 Token、position、context 和 slot，再执行模型。模型计算完后提交输出和状态，结束请求归还页。
>
> GPU 路径把本轮不等长请求片段 packed 成 N 行，线性层使用 cuBLAS，Attention 根据每行的页表和可见长度读取历史。这样既能混合 Prefill 和 Decode，也避免 CPU 路径那种逐 Token 微步组织在 GPU 上反复产生小调用。
>
> 我重点准备了两项可解释的优化。第一项是采样行裁剪：Prompt 中间 hidden 仍需计算 KV，但不需要全部投影到词表，只取真正能采样的 R 行。固定四请求实验中，词表投影总行数从 92 降到 16，激活缓冲从约 13.781 MiB 降到 2.180 MiB。端到端时间只小幅降低，所以我把显存和无用计算减少作为主要结论。
>
> 第二项是 Prefix Cache：完整公共前缀页按内容身份共享，引用计数保护活跃使用者，只有缓存保留的页可以淘汰。固定 256 Token 公共前缀实验中，目标请求 TTFT 从 miss 约 4.057 ms 降到 hit 约 1.541 ms，且生成结果与 CPU 参考一致。这个数字不包含预先建立缓存的 seed 时间。
>
> 此外项目实现了 CUDA Graph 和双 GPU PD。PD 中两卡各有完整权重与独立页池，P 生成首 Token 后，经 host 中转迁移 KV，D 从首 Token 继续执行。短 GPT-2 负载下 PD 约 18.493 ms，单卡约 9.876 ms，说明迁移和协调成本不能忽略。我的重点是能说明实现、测试和边界，而不是仅凭用了两张卡就宣称加速。

这段是完整稿，实际回答可按面试官兴趣删减。数值只需记住后面“最小数字卡”中的几组，不必同时背所有历史版本成绩。

## 4. 十分钟展开顺序

| 时间段 | 讲什么 | 建议画什么 |
| --- | --- | --- |
| 0—1 分钟 | 背景、范围、来源 | 一句话说明基于 llm.c 的扩展 |
| 1—3 分钟 | Sequence、预算、页池、Runner | `schedule → run → commit` |
| 3—5 分钟 | 一轮 packed 输入与分页 | A Decode + B Prefill 元数据表 |
| 5—7 分钟 | 一项优化深挖 | `[N,C] → [R,C] → [R,V]` 或共享页引用图 |
| 7—9 分钟 | A/B 和正确性 | baseline、指标、raw data、误差检查 |
| 9—10 分钟 | PD 负优化与 vLLM 差异 | 迁移成本、实现边界、下一步假设 |

面试官中途追问时，先回答当前问题，再回到主线。不要为了背完稿件跳过对方指出的疑点。

## 5. 一张核心数据流图

```text
请求 ID + Prompt + 生成上限
      ↓
Sequence：token_ids / computed / status / block_table
      ↓
Scheduler：本轮每请求的 scheduled_count
      ↓
BlockManager：为新位置保证物理 KV 页
      ↓
ModelInput：token / position / context / slot / 页表 / 请求边界
      ↓
GPU：Embedding → Transformer × L → LN → Gather → LM head → Argmax
      ↓
CPU commit：计算数前进 → 追加输出 → 停止判断 → 页引用归还
```

**关键代码：** [Scheduler 提交](../../mini_vllm/scheduler.hpp#L93)。

源码：[mini_vllm/scheduler.hpp，第 93—113 行](../../mini_vllm/scheduler.hpp#L93)。

```cpp
        for (std::size_t i = 0; i < output.items.size(); ++i) {
            const ScheduledItem& item = output.items[i];
            Sequence& sequence = *item.sequence;
            sequence.mark_computed(item.num_scheduled_tokens);
            block_manager_.cache_computed_prefix_blocks(sequence);
            if (sequence.pending_tokens() != 0) {
                if (sampled_token_ids[i] != -1) {
                    throw std::logic_error("partial prefill must not produce a sampled token");
                }
                continue;
            }
            const int sampled_token = sampled_token_ids[i];
            if (sampled_token < 0) {
                throw std::logic_error("completed model input requires a sampled token");
            }
            sequence.append_token(sampled_token);
            if (sequence.should_finish_after(sampled_token)) {
                sequence.set_status(SequenceStatus::Finished);
                block_manager_.release(sequence);
                finished.push_back(item.sequence);
            }
```

这段能串起三个面试问题：部分 Prefill 为什么不采样、生成 ID 为什么还没有 KV、完成请求什么时候释放页。

## 6. 高频卡 01：为什么要缓存 KV，不缓存 Q

**30 秒短答：**

> 在确定性的 causal Transformer 推理中，旧位置只能看自身及之前的输入，后续追加 Token 不改变旧位置表示，所以旧 K/V 可复用。新位置的 Query 用来读取这些历史 K/V；未来位置会生成自己的 Query，通常不再需要旧 Query。于是 KV Cache 保存跨步有用的状态，避免重复计算旧位置。

**原理展开：**

```text
第 t 步：Q_t × [K_0...K_t] → 权重 → [V_0...V_t]
第 t+1 步：新的 Q_(t+1) × [K_0...K_t,K_(t+1)]
```

**追问：旧 Token 在更深层的 hidden 也不变吗？**

回答：在模型参数、位置语义和 causal 条件固定时，可逐层归纳：第 0 层输入不变，每层旧位置只依赖旧前缀，所以输出不变。若上下文定义、模型状态或 mask 改变，需要重新检查前提。

**追问：有 KV 后 Decode 的计算量是否与历史长度无关？**

回答：不是，新 Query 仍要访问历史 KV。缓存省掉旧位置的重复投影和层计算，不删除 Attention 的历史读取。

**代码与验证：** [入门生成实验](../from_pytorch/01_generation_and_kv.md)、[CPU 实验程序](../from_pytorch/examples/attention_and_pages.py)。

## 7. 高频卡 02：生成 4 个 Token 为什么只额外计算 3 个

**短答：**

> Prompt 最后位置的 logits 已能产生第一个输出。之后每生成一个新输出，先处理上一轮生成的 ID。最后一个输出交给用户后不再继续预测，因此 Prompt 长 P、生成 G 个时，实际最多处理 P+G−1 个输入位置。

**现场例子：** P=17，G=4，最终 ID 数=21，computed=20，页大小16时最多2页。最后 `computed<total` 是正常结束状态。

**追问：EOS 恰好是首 Token 怎么办？**

回答：无需进一步 Decode；单卡释放请求页，PD 直接在 P 结束，不迁移 KV。

**证据：** [Sequence](../../mini_vllm/sequence.hpp)、[PD 首 Token 结束测试](../../dev/cuda/test_gpt2_pd_engine.cu#L109)。

## 8. 高频卡 03：分页解决什么，为什么不保证单 kernel 更快

**短答：**

> 分页通过逻辑页表管理非连续物理 KV 空间，降低对单个请求连续大块内存的依赖，并支持按需分配、回收和共享。它是存储管理方式；页表读取、离散访问和 kernel 实现仍有开销，因此不能仅凭使用分页宣称 Attention 算得更快。

**必须能写的公式：**

```text
logical_block = token_position / page_size
physical_block = block_table[logical_block]
slot = physical_block * page_size + token_position % page_size
```

**追问：碎片完全消失了吗？**

回答：没有，最后一个未填满的块仍有内部空闲槽；池容量也可能超出实际需求。当前可说明减少连续分配约束与方便复用，不说完全消除碎片。

**证据：** [CUDA 地址计算](../../mini_vllm/cuda/paged_attention.cu#L17)、[打乱物理页测试](../../dev/cuda/test_paged_attention.cu#L142)。

## 9. 高频卡 04：Packed Prefill 如何保持因果性

**短答：**

> 本轮把不同请求的已知输入片段拼成 N 行，Linear 等逐行计算使用较大的矩阵。每行仍携带所属请求的页表、绝对位置和 context length；Attention 只读自己的历史到当前位置。因此同轮先写全部新 KV，也不会读取未来位置或其他请求。

**例子：** A computed17、本轮1；B computed0、本轮3，位置 `[17,0,1,2]`，可见长度 `[18,1,2,3]`，请求边界 `[0,1,4]`。

**追问：为什么不一次并行 Decode 32 个普通 greedy 输出？**

回答：Prompt ID 已知；后续输出 ID 依赖上一轮采样，不能凭 packed 输入组织消除自回归依赖。

**证据：** [prepare_packed_model_input](../../mini_vllm/model_input.hpp#L41)、[任务 05](../task_05_multi_token_prefill_zh.md)。

## 10. 高频卡 05：采样行裁剪为什么成立

**短答：**

> 词表投影对 hidden 的各行独立。先对 N 行投影再取最后有效行，等价于先选出 R 行 hidden 再投影。项目只选择本轮完成全部输入的请求，部分 Prompt chunk 不进入 LM head；主体 Transformer 和 KV 计算仍保留。

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 763—771 行](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L763)。

```cpp
        if (config_.enable_sample_row_pruning) {
            for (std::size_t i = 0; i < output.items.size(); ++i) {
                const auto& item = output.items[i];
                if (item.sequence->num_computed_tokens() + item.num_scheduled_tokens ==
                    item.sequence->num_tokens()) {
                    last_logit_token_indices_.push_back(static_cast<int>(
                        input.query_start_locations[i + 1] - 1));
                }
            }
```

**推导：** `rows(HWᵀ) = rows(H)Wᵀ`。数学等价不意味着不同 GEMM 尺寸逐位相同，仍需要误差检查。

**追问：R=0 时是否跳过整轮？**

回答：只跳过 Gather/LM head/Argmax，KV 必须计算，否则 computed 与真实缓存不一致。

**证据：** [五种 R 回归](../../dev/cuda/test_gpt2_cuda_sample_rows.cu#L23)、[同配置 A/B](../../benchmark/results/task09_11/README.md#1-采样行裁剪)。

## 11. 高频卡 06：Prefix Cache 怎么保证共享正确

**短答：**

> 完整前缀的身份决定能否共享，不能只看当前块 Token。命中后请求页表引用已有物理页并增加 computed，跳过重复计算；引用计数保护活跃使用者。项目仅共享完整 Prompt 块，保留至少一个 Token 所在部分重新计算 logits，后续写入私有尾页。

**追问：为什么相同当前块、不同前面内容不能共享？**

回答：多层 causal 模型的 hidden 已依赖前面内容，所以 K/V 不只由当前块 ID 决定。

**追问：所有请求结束后 free 页没有全满，是否泄漏？**

回答：先看缓存是否仍持有引用。清空缓存后再核对全部归还；当前累计命中计数也不会因 clear 自动归零。

**证据：** [完整前缀 key](../../mini_vllm/block_manager.hpp#L250)、[共享引用与淘汰](../task_08_prefix_cache_zh.md)。

## 12. 高频卡 07：CUDA Graph 与 torch.compile 有什么区别

**短答：**

> CUDA Graph 复用已捕获的设备执行节点与依赖，主要减少重复的 host 提交开销；torch.compile 涉及模型图捕获、变换、降低与代码生成。两者可以组合，但缓存键、失效条件和优化目标不能混同。项目按固定 Runner 中的 N/R 缓存 CUDA Graph，动态元数据在重放前更新。

**追问：为什么行索引从 3 变 1 仍能 replay？**

回答：Graph 读固定地址的 sample_rows buffer，索引值是每轮更新的数据。N/R 变化会影响 grid/GEMM 形状，因此另查图。

**追问：改变 buffer 地址呢？**

回答：不能直接沿用旧图，图可能仍持有旧地址；需要重新捕获或满足更新机制。项目用预分配保持地址稳定。

**证据：** [graph_key 与查找](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1105)、[任务 07](../task_07_fusion_cuda_graph_zh.md)。

## 13. 高频卡 08：融合为什么会负优化

**短答：**

> 融合能减少独立 launch，也可能增加寄存器压力、同步和访存成本；端到端收益取决于原瓶颈。项目的残差加 LayerNorm 融合同时保留残差和归一化输出，并保持低精度舍入边界。使用融合开关与 Graph 开关组成四组 A/B，避免把组合收益全归给融合。

**追问：为什么仍需输出 residual？**

回答：下一次 skip connection 需要未归一化的值，只留下 normalized 会改变模型公式。

**追问：是否已证明具体退化由寄存器溢出导致？**

回答：没有，仅凭总时间不能确定这个根因。寄存器、访存、归约都属于需进一步测量的候选解释，不能把假设当成已证实结果。

**证据：** [双输出与舍入](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L391)、[阶段四组实验](../task_07_fusion_cuda_graph_zh.md)。

## 14. 高频卡 09：FP16 存储为什么还要 FP32 累加

**短答：**

> 参数、KV 和大部分 hidden 用低精度减少容量及带宽成本，归约用 FP32 降低累积误差。项目 GEMM 输入为 FP16 时指定 FP32 计算，一般 hidden 写回 FP16，最终 logits 保存 FP32；LayerNorm 和 Attention 统计也用 float。所以必须区分存储类型、累加类型和输出类型。

**追问：FP16 → float 能恢复已经丢掉的精度吗？**

回答：不能，只是后续计算使用更高表示精度。

**追问：BF16 是否已经与 FP32 生成完全一致？**

回答：没有，项目历史 BF16 实验出现 argmax 不一致，应表述为实验路径；不能把测试放宽条件后的通过当成完全等价。

**证据：** [GEMM 类型](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L998)、[测试对 BF16 的条件](../../dev/cuda/test_gpt2_cuda_model_runner.cu#L237)。

## 15. 高频卡 10：PD 交接必须恢复哪些东西

**短答：**

> 不仅要传 K/V，还要保持请求身份、Token 前缀、computed、生成参数和目标页映射一致。P 处理 Prompt 并产生首输出，首输出的 KV 还没算；D 应恢复 Prompt 的 computed，再追加这个输出，形成 pending=1 的 Decode 状态。目标页准备和迁移成功后才释放源页。

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 179—195 行](../../mini_vllm/cuda/gpt2_pd_engine.hpp#L179)。

```cpp
            pending_->transfer = p_runner_.copy_kv_to(
                d_runner_, source, *target, source.num_computed_tokens());
            // 首 Token 在 P 产生，但尚未计算它的 KV；D 从这个 Token 开始执行。
            target->mark_computed(source.num_computed_tokens());
            target->append_token(source.token_ids().back());
            pending_->source_pages = source.block_table();
            pending_->destination_pages = target->block_table();
            d_scheduler_.add(target);
        } catch (...) {
            d_blocks_.release(*target); // 迁移/接纳失败保留源状态，回收目标页。
            throw;
        }
        const auto request_id = source.request_id();
        p_blocks_.release(source);
        pending_->sequence = std::move(target);
        pending_->stage = PDStage::Decoding;
        result.handed_off.push_back(request_id);
```

**追问：两张卡页号相同才能迁移吗？**

回答：不需要，按逻辑页 i 找 source_table[i] 和 target_table[i] 分别复制。测试专门用不同物理页号和后续全词表 logits 对照验证。

**追问：属于 Tensor Parallel 吗？**

回答：不是，两端都有完整权重，拆的是推理阶段，未按矩阵维度切分参数。

**证据：** [重映射迁移测试](../../dev/cuda/test_gpt2_pd_engine.cu#L26)、[任务 11](../task_11_pd_disaggregation_zh.md)。

## 16. 高频卡 11：PD 为什么慢，还值得做吗

**短答：**

> 功能上，它验证了不同设备的独立页池、KV 迁移和请求续接。性能上，两卡不自动加速：当前采用同步 host staging，短 GPT-2 请求里传输、分配、同步和线程协调占比明显。实验中 Graph 单卡约 9.876 ms，PD 约 18.493 ms，所以我明确报告负结果。

**追问：你已经定位到哪一项是唯一根因？**

回答：迁移统计说明交接成本显著，但它包含 pinned 分配、设备切换、页拷贝与同步，不能仅靠总时间认定唯一子项。还需要更细的时间线或控制变量实验。

**追问：下一步如何优化？**

回答：先复用 staging 与持久线程，再研究传输计算重叠，并建立两卡各跑完整请求的数据并行基线；这些是待验证方案，不是当前已完成收益。

**证据：** [PD 时间与口径](../../benchmark/results/task09_11/README.md#3-双-gpu-pd-的功能与成本)、[实际重叠摘要](../../benchmark/results/task09_11/nsys_overlap.json)。

## 17. 高频卡 12：正确性为什么不只比较生成文本

**短答：**

> 文本或 greedy ID 相同只能说明最大项没有改变，不能保证其余 logits 和 KV 正确。项目用独立 CPU 完整前缀参考、全词表误差、打乱页映射、跨页边界、请求回收和内存检查组合验证。不同测试证明不同不变量，不能用一个 PASS 替代全部证据。

**追问：两个 GPU 路径相互对比可能有什么问题？**

回答：如果共享同一个错误，两者可能一起错，所以还需要实现方式不同的独立参考。

**追问：memcheck 通过就数值正确吗？**

回答：不是，它主要检查访存问题；错用合法地址、错误 mask 或错误样本映射仍可能无非法访存。

**证据：** [完整验证记录](../../benchmark/results/task09_11/validation.txt)、[内存检查](../../benchmark/results/task09_11/sanitizer_pd.txt)。

## 18. 面试前只记这张最小数字卡

以下全部来自 [2026-09-11 同轮证据](../../benchmark/results/task09_11/README.md)。不同实验负载不同，不能横向拼出累计加速倍数。

| 场景 | 推荐记的数字 | 随口必须补充的条件 |
| --- | --- | --- |
| 采样行 | 92 → 16 行 | Prompt 8/16/24/32，各生成4 |
| 激活缓冲 | 13.781 → 2.180 MiB | FP16、固定容量；不是整卡显存 |
| 裁剪 Graph 总时间 | 5.314 → 5.227 ms | 7 次中位数，约1.7%，差异较小 |
| 256 前缀 TTFT | 4.057 → 1.541 ms | miss→hit，seed 在目标计时外，约62%下降 |
| 双卡 PD Graph | 9.876 → 18.493 ms | 单卡→PD，Prompt17/33/49，各输出8 |
| PD 逻辑载荷 | 5.0625 MiB/整批 | 9页，D2H+H2D 总链路字节约两倍 |

不必强记更多小数位。说清对象、单位、基线和边界，比把数字背到小数点后六位更有价值。

## 19. 简历句子对应哪份证据

| 可使用的能力表述 | 实现入口 | 验证材料 | 表述边界 |
| --- | --- | --- | --- |
| 分页 KV 与连续批处理 | [BlockManager](../../mini_vllm/block_manager.hpp)、[Scheduler](../../mini_vllm/scheduler.hpp) | [控制面测试](../../dev/test_mini_vllm_control_plane.cpp) | 无活跃请求抢占 |
| packed Prefill + GPU Runner | [ModelInput](../../mini_vllm/model_input.hpp)、[Runner](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu) | [GPU 测试](../../dev/cuda/test_gpt2_cuda_model_runner.cu) | GPT-2 教学模型范围 |
| 只投影采样行 | [Gather](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L638) | [采样行回归](../../dev/cuda/test_gpt2_cuda_sample_rows.cu) | 不删除主体 KV 计算 |
| Prefix 共享与性能测量 | [缓存管理](../../mini_vllm/block_manager.hpp#L101) | [prefix.csv](../../benchmark/results/task09_11/prefix.csv) | 人工构造命中，非自然线上分布 |
| CUDA Graph 动态元数据 | [graph_key](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1105) | [固定 N/不同 R 测试](../../dev/cuda/test_gpt2_cuda_sample_rows.cu) | 精确形状缓存，无自动淘汰 |
| 功能性双卡 PD | [PDEngine](../../mini_vllm/cuda/gpt2_pd_engine.hpp) | [迁移与流水测试](../../dev/cuda/test_gpt2_pd_engine.cu) | host staging，短负载未加速 |

仓库较早的 `doc/resume_project.tex` 包含旧阶段数字；准备当前回答时，优先使用上表对应的最新证据，不混用不同版本条件。本轮不自动改动你的个人完整简历。

## 20. 七种容易答过头的说法

| 不准确说法 | 更准确的表达 |
| --- | --- |
| 我完整复现了 vLLM | 项目实现了若干核心抽象，并对照固定版本源码 |
| 分页消灭显存碎片且必然加速 | 改善分配与复用，尾块仍有空闲，速度取决于实现 |
| 两张卡实现了模型切分 | 当前 PD 两卡各有完整模型，按阶段分工 |
| FP16 所有计算都是 half | 主要存储低精度，多个归约与 logits 使用 float |
| Graph 让计算量下降 | 主要减少重复提交开销，算子数学工作量未自动减少 |
| kernel 数减少证明端到端更快 | 还需同条件的墙钟时间与正确性验证 |
| 吞吐提高 62% | 该记录是指定前缀下 TTFT 下降约62%，不是同一指标 |

## 21. 由浅到深的模拟追问串

**线索 A：从分页追到正确性。**

```text
为什么分页？
→ 写出 slot 公式
→ 页表 [5,2] 的位置 17 在哪里？
→ 释放后重用同一页会不会看到旧值？
→ 新 KV 写入与 context 怎样一起防止读无效槽？
→ 如何用打乱映射和全词表对照测试？
```

**线索 B：从优化追到实验。**

```text
采样行怎么省？
→ 为什么只能裁剪末尾投影？
→ R=0 怎么办？
→ Graph 缓存键要不要变？
→ 为什么显存降很多，时间只降一点？
→ 这个小幅收益怎样与噪声区分？
```

**线索 C：从多卡追到系统边界。**

```text
PD 与 TP 区别？
→ 首 Token 哪端生成？
→ 目标 computed 恢复多少？
→ 两端物理页号不同怎么办？
→ D 满了如何背压？
→ 为什么目前比单卡慢，下一步先测什么？
```

每条线练三遍：第一遍看答案，第二遍只看问题，第三遍由自己改变数字再回答。改变数字后仍能推导，才说明记住了因果关系。

## 22. 自评分与记忆方法

每题记 0—3 分：0 分只能报术语；1 分能说结论；2 分能解释原因并手算；3 分能指出代码、验证和边界。优先把核心卡从 1 分提高到 2 分，再追求所有代码行号。

推荐一次 25 分钟：前 5 分钟复述上次两题，中间 15 分钟学两张新卡，最后 5 分钟写一个公式和一个测试断言。第二天先闭卷回答，发现混淆后只回读对应任务小节。

需要背的稳定内容：因果关系、公式、状态机、实验口径。适合查表的内容：长路径、精确行号、完整编译命令和不常用参数。

完成本篇后，进入后三本手册，用少量编码、定位和分析练习检查这些回答能否落到真实数据上。
