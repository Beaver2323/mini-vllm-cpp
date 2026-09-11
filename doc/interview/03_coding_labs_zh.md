# 面试手册 03：四个代码补全实验

[面试学习目录](README.md) · 上一篇：[口述与证据](02_project_defense_zh.md) · 下一篇：[故障定位](04_debug_cases_zh.md)

这四个实验分别检验记账、地址、行索引和 PD 恢复。每个实验只补一个小函数，直接使用项目的 Sequence/SchedulerOutput 类型，不增加引擎功能。

建议顺序：先口述 30 秒答案，再写 5—15 行代码，运行检查，最后解释一个失败边界。第一次可以看答案，第二次只看函数签名，第三次换一组数字闭卷完成。

## 1. 文件、构建和预期结果

| 文件 | 作用 |
| --- | --- |
| [lab_tasks.hpp](examples/lab_tasks.hpp) | 学习者补全的四个 TODO |
| [lab_solutions.hpp](examples/lab_solutions.hpp) | 参考实现，按实验逐个看 |
| [lab_checks.cpp](examples/lab_checks.cpp) | 检查真实状态、页映射和输入组织 |
| [verified_output.txt](examples/verified_output.txt) | 本次已验证的参考输出 |

建议复制题目到 `/tmp` 再练习，仓库原题和答案可一直保留：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
mkdir -p /tmp/zyf_interview_practice
cp doc/interview/examples/lab_tasks.hpp /tmp/zyf_interview_practice/
cp doc/interview/examples/lab_checks.cpp /tmp/zyf_interview_practice/
c++ -std=c++17 -O0 -g -gdwarf-4 -I. \
  /tmp/zyf_interview_practice/lab_checks.cpp -o /tmp/zyf_interview_practice/check
/tmp/zyf_interview_practice/check 1
```

初始题目会输出 `TODO 1: sample_ready` 并非零退出，这是题目尚未完成的预期结果。编辑 `/tmp/zyf_interview_practice/lab_tasks.hpp` 后需要重新执行编译命令；运行旧二进制不会自动反映修改。

只想先验证参考答案：

```bash
c++ -std=c++17 -O0 -g -gdwarf-4 -I. -DLAB_USE_SOLUTION \
  doc/interview/examples/lab_checks.cpp -o /tmp/zyf_interview_lab_answers
/tmp/zyf_interview_lab_answers all
```

参考输出：

```text
L1 PASS: partial=false final=true decode=true computed=5 total=6
L2 PASS: positions=[15,16,17,32] slots=[95,32,33,144]
L3 PASS: positions=[17,0,1,2] rows=[0,3] partial_rows=[0] R0=[]
L4 PASS: handoff=17/18 first_decode=18/19 invalid_prefix=rejected
```

这些实验只需要 C++17 编译器，不下载模型、不访问 GPU。L4 只检查交接后的 CPU 记账，真正的 KV 迁移仍由项目 PD GPU 测试验证。

## 2. 实验 L1：本轮能否采样

**面试问题：** Prompt 分成多个 chunk 后，什么时候才能输出第一个 Token？

**先背一句：** 只有本轮计算完所有已知输入，才用最后位置的 logits 续写；中间 chunk 返回无采样标记。

打开 [Sequence::pending_tokens](../../mini_vllm/sequence.hpp#L46) 与 [Runner 的采样资格](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L763)，再写：

```cpp
bool sample_ready(const Sequence& sequence, std::size_t scheduled);
```

**契约：** `scheduled` 必须大于 0 且不超过 pending；非法时抛异常。合法时判断本轮结束后 pending 是否为 0。函数本身不修改 Sequence。

先填这张表：

| computed | total | scheduled | 是否可采样 | 原因 |
| ---: | ---: | ---: | --- | --- |
| 0 | 5 | 3 | ? | Prompt 尚余多少 |
| 3 | 5 | 2 | ? | 是否到已知输入末尾 |
| 5 | 6 | 1 | ? | 生成 ID 作为本轮输入 |
| 5 | 6 | 2 | ? | 是否超出实际输入 |

**参考答案：** false、true、true、抛异常。

实现中比较 `scheduled == pending_tokens()`，比先做可能溢出的加法再比较 total 更直接。不要用 `is_prefill()` 判断能否采样，最后一个 Prefill chunk 同样能产生输出。

**上下游调用点：**

```text
Scheduler 给出 scheduled
  → Runner 判断 sample_ready
  → 有资格则选末行 logits
  → Scheduler::commit 先 mark_computed，再 append_token
```

**常见错误：**

```cpp
// 错误教学例：所有 Prefill 都不采样，会让首 Token 永远无法产生。
return !sequence.is_prefill();
```

它忽略了“Prompt 最后一块产生首输出”。本项目阶段与采样资格是不同概念。

**检查命令：** `/tmp/zyf_interview_practice/check 1`。

**代码位置：** 参考答案 `lab_solutions.hpp` 中的 `sample_ready`；检查程序 `lab_checks.cpp` 的 `lab1`。

**面试追问：** 为什么 G 个输出只计算 G−1 个新输入？

参考回答：Prompt 末行先预测第一个输出，最后一个输出不再喂回模型；所以计算进度与 ID 数不能强行相等。

**闭卷变式：** computed=15,total=18，本轮2 → 不采样；本轮3 → 采样。答案不能依赖具体页大小。

## 3. 实验 L2：从逻辑位置算物理槽

**面试问题：** 物理页不连续时，第 17 个位置如何写 KV？

**先背一句：** 先把位置除以页大小得到逻辑块，再经页表找物理块，最后加页内偏移。

函数：

```cpp
std::size_t physical_slot(const std::vector<int>& table,
                          std::size_t position,
                          std::size_t page_size);
```

**契约：** 页大小为正；逻辑块在表内；对应物理页非负。本教学函数不持有页池总容量，物理页是否超过实际池大小由调用方进一步检查；生产 ModelInput 有相应检查。

真实上游在 [prepare_packed_model_input](../../mini_vllm/model_input.hpp#L71)，下游在 [write_kv_cache_kernel](../../mini_vllm/cuda/paged_attention.cu#L69)。

给定 `table=[5,2,9]`、page_size=16：

| position | 逻辑块 | 物理块 | offset | slot |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 0 | 5 | 15 | 95 |
| 16 | 1 | 2 | 0 | 32 |
| 17 | 1 | 2 | 1 | 33 |
| 32 | 2 | 9 | 0 | 144 |

**参考推导：** `slot = table[position/16]*16 + position%16`。这是元素槽位，不是字节偏移；真正的 KV 数组还要乘层、头和 head_dim 的 stride。

**为什么测试要打乱页号：** 若页表恰好为 `[0,1,2]`，错误公式 `slot=position` 也能通过，测试无法区分正确映射与侥幸一致。

**三种非法输入：** page_size=0；position=48 时访问第 3 号逻辑块但表长只有3；表中页号为 -1。检查应先于除法或数组读取。

**检查命令：** `/tmp/zyf_interview_practice/check 2`。

**追问：位置 Embedding 用 slot=33 还是 position=17？**

回答：position=17。slot 是物理缓存地址，与模型中的绝对位置语义不同。

**闭卷变式：** table=[7,3]，position=31 → slot=63；position=32 → 表容量不足，不能继续假设物理页连续。

## 4. 实验 L3：三种行号之间建立映射

**面试问题：** 为什么模型 packed 行数、请求数和采样行数可能都不同？

**先背一句：** 请求贡献的本轮输入长度不同，N 是长度之和；每个完成输入的请求最多产生一个样本，R 只统计这些请求。

函数：

```cpp
std::vector<std::size_t> sample_rows(
    const SchedulerOutput& output,
    const std::vector<std::size_t>& starts);
```

**契约：** starts 从0开始，长度为请求数+1，最后为总 Token 数，每段长度与 item 的 scheduled 相等；只选 L1 判定为可采样的片段末行。

真实检查程序先使用项目的 `BlockManager` 为请求分配页，再调用真实 `prepare_packed_model_input`，然后把产生的 `query_start_locations` 送入你补全的函数。

```text
请求 A：Prompt 17 已算完，另有一个刚生成 ID 待执行
请求 B：新 Prompt 3 个 Token
本轮：A 1 行 + B 3 行
positions = [17,0,1,2]
starts = [0,1,4]
可采样 rows = [0,3]
```

接着把 B 本轮数从3改为2：

```text
starts = [0,1,3]
A 处理完已有输入 → 行0可采样
B 只处理 Prompt 前2个 → 行2不可采样
rows = [0]
```

最后只调度 B 的前2个 Token：R=0，返回空向量，仍是合法执行计划。

**参考算法：**

```cpp
// 教学伪代码；完整边界检查见 lab_solutions.hpp。
for each item i:
    validate(starts[i+1] - starts[i] == item.scheduled)
    if sample_ready(item.sequence, item.scheduled):
        rows.push_back(starts[i+1] - 1)
```

**三个索引要分别解释：**

| 索引 | 作用 | 示例 |
| --- | --- | --- |
| item_index | 调度中的请求序号 | B 是1 |
| packed_row | 本轮所有输入拼接后的行号 | B 末行是3 |
| sample_index | 紧凑 logits/Token 数组中的行号 | B 样本是1 |

**常见错误：** 不检查资格，直接 `starts[1:]-1`。它会把 partial Prompt 的末行也当成对外样本。本项目任务09要求在 LM head 前过滤；固定 vLLM 版本的另一条处理路径见手册01，不要混同。

**检查命令：** `/tmp/zyf_interview_practice/check 3`。

**追问：选出的 hidden 行次序能否随意排序？**

回答：可以设计不同顺序，但必须同步维护样本到请求的反向映射。当前实现按调度顺序生成 rows 并按同顺序还原，不能单独改其中一端。

**闭卷变式：** 长度 `[2,3,1]`，只有前两个请求完成输入，starts=`[0,2,5,6]`，rows=`[1,4]`。

## 5. 实验 L4：PD 交接后恢复 CPU 状态

**面试问题：** P 端产生首 Token 后，D 端 computed 应等于多少？

**先背一句：** computed 等于已迁移 KV 的 Prompt 长度；首输出只是 ID，还需要 D 执行它的前向。

函数：

```cpp
void restore_after_handoff(Sequence& target, const Sequence& source);
```

**本实验前提：** KV 已由外部迁移，源目标模型和采样参数一致；这里只恢复 CPU 计数。函数本身不验证 GPU 内存，也不执行 `copy_kv_to`。

有效 source：Prompt=17、computed=17、total=18，只有一个输出42。有效 target：相同请求 ID、相同 Prompt，computed=0、total=17。

需要在修改 target 前检查：

- request ID 一致。
- 源计算量等于 Prompt 长度，并且恰有一个生成 ID。
- 目标尚未计算，长度与源 Prompt 相同。
- Prompt Token 内容一致。

然后仅做两步：

```cpp
// 这两句也是生产交接路径中的核心顺序。
target.mark_computed(source.num_computed_tokens());
target.append_token(source.token_ids().back());
```

结果：target computed17,total18,pending1。再模拟一次 D 模型输入完成、追加新输出43，得到 computed18,total19。

**错误答案：** 直接 `mark_computed(source.num_tokens())`。目标只有17个 Prompt ID，却试图标记18个已算，真实 Sequence 会拒绝；若绕过检查，也会把未执行的首输出当成已有 KV。

**调用位置：** [try_handoff](../../mini_vllm/cuda/gpt2_pd_engine.hpp#L169)。生产代码还负责目标页预留、真正迁移、加入 D Scheduler、错误清理和源页释放，不能把本实验函数当成完整 PD 实现。

**检查命令：** `/tmp/zyf_interview_practice/check 4`。

**追问：为什么恢复前先校验全部前提？**

回答：避免修改一半计数后才发现 Prompt 不一致，让调用方不知道目标处于什么状态。本练习只保证这些前提检查在状态变更前完成，不声称真实 CUDA 迁移具备完整事务恢复。

**闭卷变式：** Prompt32，P 产生第一个输出后 D 仍从 computed32,total33 开始；是否多分配一页由下一输入位置与页容量决定，不改变这个记账关系。

## 6. 用断点检查自己的答案

使用前面的 `-O0 -g -gdwarf-4` 编译结果：

```bash
gdb /tmp/zyf_interview_lab_answers
```

```text
(gdb) break interview_lab::restore_after_handoff
(gdb) run 4
(gdb) next
(gdb) print prompt
(gdb) list
```

如果停在函数入口，先 `next` 到局部变量初始化之后再观察。结构私有字段可直接在 GDB 查看，例如 `target.num_computed_tokens_`，不必依赖调用被内联的访问器。

更稳定的做法是先 `rg -n 'target.mark_computed|target.append_token' doc/interview/examples/lab_solutions.hpp`，再按实际行号在两句前后停下。具体命令执行记录放在 [gdb_verified_output.txt](examples/gdb_verified_output.txt)；不同 GDB 版本的排版可能不同。

## 7. 怎么知道自己不是背答案

四个实验分别通过后，再做这些口头变化，不急着增加第五个功能：

| 实验 | 改一个条件 | 应能立即解释的变化 |
| --- | --- | --- |
| L1 | Prompt 更长但预算不变 | 更多轮 partial chunk，首 Token 更晚 |
| L2 | 随机打乱物理页号 | 逻辑 position 不变，slot 改变 |
| L3 | 一个请求晚一轮结束 Prompt | N 未必变，R 和样本映射改变 |
| L4 | max_new_tokens=1 | 生产 PD 在 P 端直接结束，不需要交接 |

面试时可这样介绍自己的理解：

> 我把请求状态、页映射和样本映射拆成小型可验证问题。能够手算输入和输出，再用真实项目类型检查边界。这样即使调度顺序或页号变化，也能从不变量解释，而不是只记住某一次运行结果。

本篇参考答案和检查程序只覆盖明确列出的教学契约。生产规模下的整数上界、异常恢复、并发访问和任意模型适配，需要额外设计，不能由四个小实验的 PASS 推导出来。
