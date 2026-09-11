# 面试手册 04：四个可复现的故障定位案例

[面试学习目录](README.md) · 上一篇：[代码补全](03_coding_labs_zh.md) · 下一篇：[性能分析实操](05_profiling_lab_zh.md)

本篇四个错误均为**人为设计的教学案例**，位于独立示例程序，没有把错误写入生产引擎，也不代表它们都曾是项目历史缺陷。它们用于练习如何从现象找到第一个状态或数据分歧。

案例 F1/F2/F4 使用实际项目类型或输入构造；F3 用 CPU map 模拟动态元数据被误缓存，没有执行错误的 CUDA Graph。真实 GPU 回归入口会在每个案例中分别指出。

## 1. 先记住定位问题的六步

> 固定输入和配置，找到最早不同的位置，沿生产者与消费者核对语义，构造最小反例，修正原因，再用反例与原场景回归。

| 步骤 | 你要问的问题 | 本项目观察量 |
| --- | --- | --- |
| 固定场景 | 同一模型、精度、Prompt、预算吗？ | 配置、Token ID、开关 |
| 分类 | 状态错误、地址错误、数值错误还是计时问题？ | pending、slot、logits、时间边界 |
| 找首个分歧 | 第一个错误出现在哪轮哪行？ | request_id、position、packed row |
| 查上下游 | 谁写了这个值，谁按什么含义读取？ | ModelInput → kernel / Runner → commit |
| 最小反例 | 什么最小输入仍能触发？ | 跨页16/17、两请求、同N不同R |
| 回归 | 修复是否同时保住正常路径？ | 非连续页、部分chunk、清理后free |

“最终 Token 错了”只说明发生过分歧，不足以证明 Attention kernel 有 bug。先比较输入元数据，可以省掉很多盲目逐层打印。

## 2. 编译并观察四个教学错误

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
c++ -std=c++17 -O0 -g -gdwarf-4 -I. \
  doc/interview/examples/fault_cases.cpp -o /tmp/zyf_interview_faults
/tmp/zyf_interview_faults
```

预期输出：

```text
F1 已检测: position=17 错误slot=17 正确slot=33
F2 已检测: 错误positions=[0,0,1,2] 正确=[17,0,1,2]
F3 已检测: 同(N,R)=(4,1) 旧rows=[3] 当前必须=[1]
F4 已检测: 错误computed/total=18/18 正确=17/18 pending必须为1
四个教学错误均被定位条件检测；生产引擎未修改。
```

这里程序成功退出表示四个错误都被检查条件识别，不表示错误算法通过了正确性验证。如果“错误版本”和期望没有区别，程序反而会失败，提示教学用例失去检测力。

## 3. F1：绕过页表，逻辑位置被当成物理槽

**假设现象：** 顺序页分配时正常，打乱或复用页后输出才出错；memcheck 可能仍无非法访存。

**最小反例：** 页大小16，页表 `[5,2]`，position17。正确 slot33，错误实现返回17。

源码：[doc/interview/examples/fault_cases.cpp，第 14—19 行](../../doc/interview/examples/fault_cases.cpp#L14)。

```cpp
        // F1：绕过页表，把逻辑位置当作物理 slot。
        const std::vector<int> table{5,2};
        const auto expected_slot=interview_lab::physical_slot(table,17,16);
        const auto bad_slot=(17/16)*16+17%16;
        detected(expected_slot!=static_cast<std::size_t>(bad_slot),
                 "F1 已检测: position=17 错误slot=17 正确slot=33");
```

错误公式实际上把 `table[logical_block]` 省掉了。逻辑块1对应物理页2，不是物理页1。

**如何定位：**

1. 在 [ModelInput 的 physical_block 计算](../../mini_vllm/model_input.hpp#L71) 停下。
2. 看请求页表、position、physical_block、最终 slot。
3. 再看 [write_kv_cache_kernel](../../mini_vllm/cuda/paged_attention.cu#L69)，它会把 slot 拆回物理页与页内位置。
4. 两端都同意“slot 是物理 Token 槽”后，再追层/头/维度 stride。

**为什么不一定崩溃：** 17 可能仍是合法池内地址，只是属于错误页或请求。地址合法与语义正确是两种检查。

**修复原则：** 使用 `block_table[position/page_size]` 得到物理块，并检查逻辑块存在、物理块在实际池容量内。

**回归要求：** 顺序页、打乱页、15/16/17边界、页回收复用都要包含。真实 GPU 独立测试见 [test_paged_attention.cu](../../dev/cuda/test_paged_attention.cu)。

**面试 30 秒复述：**

> 我会先用非连续页表构造反例，检查逻辑位置到物理槽的转换。如果错误地址仍在池内，memcheck 不一定发现，所以需要页映射与数值参考一起验证。修复的关键是保留页表间接寻址，而不是只修某个越界位置。

**追问：** 如果测试页表只有 `[0,1,2]`，为什么容易漏？

答案：此时物理页号恰等于逻辑页号，错误简化碰巧与正确公式一致。

## 4. F2：混合 Batch 时已有请求的位置被归零

**假设现象：** 每个请求单独跑正常，Decode 与新 Prefill 混合时已有请求出错。

**最小反例：** A 已 computed17、有一个生成 ID 待算；B 是新 Prompt3。正确 positions=`[17,0,1,2]`，错误用局部 offset 得到 `[0,0,1,2]`。

源码：[doc/interview/examples/fault_cases.cpp，第 21—30 行](../../doc/interview/examples/fault_cases.cpp#L21)。

```cpp
        BlockManager blocks(8,16);
        auto a=std::make_shared<Sequence>(1,std::vector<int>(17,10),SamplingParams{4,-1,true});
        auto b=std::make_shared<Sequence>(2,std::vector<int>{20,21,22},SamplingParams{4,-1,true});
        a->mark_computed(17);a->append_token(30);
        if(!blocks.ensure_capacity(*a,18)||!blocks.ensure_capacity(*b,3))throw std::runtime_error("allocation");
        SchedulerOutput output{{{a,ExecutionPhase::Decode,1},{b,ExecutionPhase::Prefill,3}},4};
        const auto input=prepare_packed_model_input(output,blocks,64,4,8);
        const std::vector<int> wrong_positions{0,0,1,2};
        detected(input.positions!=wrong_positions,
                 "F2 已检测: 错误positions=[0,0,1,2] 正确=[17,0,1,2]");
```

这里 `prepare_packed_model_input` 是真实项目函数，错误 vector 是专门构造的对照。运行结果证明实际函数保留了 A 的历史位置，不是在 GPU 里故意制造越界。

**定位时记录一整行契约：**

| 行 | request | position | context | 含义 |
| ---: | --- | ---: | ---: | --- |
| 0 | A | 17 | 18 | 老请求下一次 Decode |
| 1 | B | 0 | 1 | 新 Prompt 的首位置 |
| 2 | B | 1 | 2 | 新 Prompt 的第二位置 |
| 3 | B | 2 | 3 | 新 Prompt 的末位置 |

**根因：** 混淆了本轮局部 offset 与请求绝对位置。正确关系是 `position=sequence.computed+offset`。

**下游影响：** GPT-2 的位置 Embedding 会读错位置；context 可能变短；slot 可能覆盖早期 KV。最终 logits 出错只是多个错误传播后的表现。

**GDB 实操：**

```text
(gdb) break doc/interview/examples/fault_cases.cpp:28
(gdb) run
(gdb) print input.positions
(gdb) print input.context_lengths
(gdb) print input.query_start_locations
```

预期分别是 `{17,0,1,2}`、`{18,1,2,3}`、`{0,1,4}`。断在第28行时 `input` 已构造，避免在变量还未初始化时读取。

**修复后的检查：** 除 positions 外，还要检查 request_ids 和 scheduled_item_indices，避免位置正确但样本被交给错误请求。

**真实回归入口：** [GPU Runner 的混合与复用测试](../../dev/cuda/test_gpt2_cuda_model_runner.cu#L200)。

**30 秒复述：**

> 混合批次问题优先检查输入组织。packed 行号和局部 offset 每轮变化，绝对 position 要加上每个请求自己的 computed。先把请求、位置、context 和 slot 放在同一张表里比较，再进入模型层定位，通常更容易找到首个分歧。

## 5. F3：Graph 形状相同，动态行号仍需更新

**假设现象：** 首轮正确，后续相同 N/R 但请求组合改变时，输出似乎来自旧行。

本案例用 map 模拟错误缓存行为：把本应每轮更新的采样行值，误当成了可重复使用的静态内容。

源码：[doc/interview/examples/fault_cases.cpp，第 32—39 行](../../doc/interview/examples/fault_cases.cpp#L32)。

```cpp
        // F3：用 map 模拟把动态采样行误当成静态图缓存内容。
        // 此处没有运行真正 CUDA Graph，检验的是元数据必须更新的条件。
        std::map<std::pair<int,int>,std::vector<int>> bad_cached_rows;
        const auto key=std::make_pair(4,1);
        bad_cached_rows.emplace(key,std::vector<int>{3});
        const std::vector<int> current_rows{1};
        detected(bad_cached_rows.at(key)!=current_rows,
                 "F3 已检测: 同(N,R)=(4,1) 旧rows=[3] 当前必须=[1]");
```

第一次 `(N=4,R=1)` 采样行3；第二次同形状应采样行1。形状没变，只能说明 grid/GEMM 规模可复用，不能说明输入数据相同。

**真正 CUDA 路径应检查三件事：**

1. [run](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L762) 是否重新生成 `last_logit_token_indices_`。
2. [forward](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L1101) 是否把新行号上传到 sample_rows 设备缓冲。
3. 元数据 copy 与 Graph launch 是否在有依赖的顺序中，图中指针是否仍指向这个缓冲。

**另一个相邻但不同的问题：** 如果 N相同、R变化，项目需要另一张图；这属于形状键问题。当前案例 N/R都相同，只讨论动态数据刷新。两个错误不要混成一个。

**修复原则：** 缓存设备执行计划和固定地址，每轮更新 Token、页表、长度、采样行等数据内容。

**真实回归：** [采样行测试](../../dev/cuda/test_gpt2_cuda_sample_rows.cu#L23) 使用 R=0/1/2/1/0，且两次 R=1 的行号不同。它实际运行 CUDA Graph 与完整 logits 对照；本教学 map 本身没有这个证明能力。

**30 秒复述：**

> Graph 能复用的是形状和地址满足条件的执行计划。相同 N/R 下，Token、页号和采样行仍可能变化，我会检查这些数据是否在 replay 前更新到图读取的设备缓冲。调试要同时区分图缓存键错误和动态内容没有刷新的错误。

**追问：** 为什么简单地对每个请求新建一张图不是理想修复？

答案：那可能掩盖元数据协议错误，同时增加捕获与缓存成本；应先证明固定计划读取新数据的路径正确。

## 6. F4：PD 误把首输出当成已经计算 KV

**假设现象：** P 输出首 Token 正常，D 接管后无待算输入、跳过一次执行或后续结果出错。

源码：[doc/interview/examples/fault_cases.cpp，第 41—48 行](../../doc/interview/examples/fault_cases.cpp#L41)。

```cpp
        Sequence source(7,std::vector<int>(17,10),{4,-1,true});
        Sequence target(7,std::vector<int>(17,10),{4,-1,true});
        source.mark_computed(17);source.append_token(42);
        interview_lab::restore_after_handoff(target,source);
        const auto expected_pending=target.pending_tokens();
        target.mark_computed(1); // 故意错误：并没有执行首输出的模型前向。
        detected(target.pending_tokens()!=expected_pending,
                 "F4 已检测: 错误computed/total=18/18 正确=17/18 pending必须为1");
```

参考恢复得到 target=`17/18`；错误代码又调用 `mark_computed(1)`，却没有执行这个输入的模型前向，使状态变成`18/18`。

**关键变量：**

```text
source.num_prompt_tokens = 17
source.num_computed_tokens = 17
source.num_tokens = 18
target 正确恢复：computed=17, total=18, pending=1
D 第一输入：source 最后追加的生成 ID
```

**为什么不能只检查 token_ids：** 错误版本仍可能保存完全相同的 ID 数组，但计算进度与 KV 内容已不一致。请求状态正确性需要多个字段联合约束。

**真实调用点：** [try_handoff](../../mini_vllm/cuda/gpt2_pd_engine.hpp#L179) 先复制已计算 KV，再 mark source 的 computed，再 append 首输出；不是把 source 的全部 Token 数当作计算量。

**修复后回归：** Prompt1/16/17/31/32/33；首输出即EOS；生成数1；源目标页号不同；D容量受限；所有页归还。真实测试见 [test_gpt2_pd_engine.cu](../../dev/cuda/test_gpt2_pd_engine.cu)。

**30 秒复述：**

> PD 迁移的是已经计算的 KV，P 刚产生的首输出只有 ID。D 应恢复 Prompt 的 computed，再追加这个 ID，形成一个待执行 Token。如果 computed 被设成包含首输出的总长度，就会把没有 KV 的位置当成已完成。要同时验证 Token、计数、页映射和续写 logits。

## 7. 怎样区分真实历史记录与教学案例

| 材料 | 可以怎么介绍 | 不能据此声称什么 |
| --- | --- | --- |
| 本篇 F1—F4 | 为理解不变量构造并运行了小型错误案例 | 四个问题都曾发生在线上或由本人独立修复 |
| GPU 回归测试与日志 | 项目验证过这些场景与精度条件 | 测试覆盖全部模型和并发配置 |
| Fusion/PD 负优化记录 | 在固定负载观察到对应性能结果 | 未测过的寄存器或链路因素就是唯一原因 |
| 个人实习经历 | 按实际参与和可公开信息说明定位过程 | 把本篇练习替换成公司真实缺陷经历 |

面试官问“你修过什么问题”时，先说明这是项目验证、教学反例还是实际修复，再讲方法。具体、可核查的回答比夸大经历更经得起追问。

## 8. 运行问题先排除这些基础条件

这里整理的是执行本仓库学习命令的常见定位入口，不是声称本机当前都存在这些故障。

| 现象 | 先执行 / 检查 | 解释 |
| --- | --- | --- |
| 找不到头文件 | `pwd`，确认在 `llm.c` 根目录并带 `-I.` | 工作目录决定相对 include |
| 源码改了输出没变 | 重新编译，确认运行 `/tmp` 中对应二进制 | C++ 不会自动重编译 |
| Python 版本不对 | `conda run -p /home/miniconda3/envs/zyf1 python --version` | 明确使用 zyf1 |
| GPU 测试找不到 checkpoint | 检查根目录 `gpt2_124M.bin` | CPU 小实验无需这个文件 |
| PD 只看到一张卡 | 检查 `CUDA_VISIBLE_DEVICES` | 单卡可见配置不能运行双卡测试 |
| GDB 变量不可读 | `-O0 -g -gdwarf-4`，停在变量初始化后 | 优化和调试格式影响观察 |
| GDB 不能启动子进程 | 查看是否被 ptrace 权限限制 | 属于运行环境，不能据此推断代码错误 |

这里没有要求为源码阅读安装一整套 vLLM 依赖。固定版本对照直接用链接和函数位置，C++ 实验独立运行。

## 9. 从日志进入 CUDA 数值定位

如果真实 GPU 测试失败，建议保存：请求 ID、轮次、position、context、页表、slot、采样行和第一处 logits 差异。随后按层比较 Embedding、LN、QKV、Attention、MLP。

精度对比要尽量固定同一个 Token 前缀。自由生成路径一旦某步 argmax 分叉，后续输入已不同，不能把逐步变大的 logits 差异直接归因于某个固定算子。

Compute Sanitizer 检查非法访存，GDB/日志检查 CPU 状态，reference 检查数学结果，Nsight 检查执行时间。四者分工不同；不应仅凭一个工具没有报错就停止排查。

## 10. 闭卷演练题

**题1：只有页复用后错，首先怀疑什么？**

答案：请求页表与 slot 映射、旧引用是否已释放、可见长度是否读入无效尾槽。先验证这些不变量，再定位数值 kernel。

**题2：只有 mixed batch 错，如何快速缩小？**

答案：固定两请求，一条 Decode 一条短 Prefill，打印每行 request/position/context/slot/qstart；把模型执行前的元数据与手算表对照。

**题3：同 N/R 第二次 replay 错，是不是一定缓存键缺字段？**

答案：不一定，也可能动态值没有刷新或地址生命周期错误。分别验证形状与数据协议。

**题4：PD 两边 ID 一样，能证明迁移成功吗？**

答案：不能，computed、目标页映射和 KV 数值仍可能错误。还需续写全词表对照和页回收检查。

**题5：如何用一分钟总结自己的调试方法？**

参考回答：先固定输入与执行配置，按状态、地址、数值分层找第一个分歧；用最小边界反例定位生产者和消费者之间的契约；修复后同时回归错误场景和正常路径，最后再看性能是否受影响。
