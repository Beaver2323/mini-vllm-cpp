# 第 6 节：实验、断点、术语与自测答案

上一节：[nano-vLLM 与 vLLM](05_read_nanovllm_and_vllm.md) · [返回目录](README.md)

本篇用来实际操作。每次先写预测，再运行，再解释输出。前三个实验只用 CPU，不要求安装
vLLM、下载模型或接触双卡；后面的项目入口逐步增加复杂度。

## 1. 实验分级与运行环境

从仓库根目录运行：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda activate zyf1
```

| 级别 | 实验 | 需要什么 | 验证什么 |
| --- | --- | --- | --- |
| A | PyTorch 缓存、分页、Packed | zyf1 现有 PyTorch，CPU | 数学与索引等价性 |
| B | C++ 控制面 demo | C++17 编译器 | 真正 Scheduler 的队列/计数/页回收 |
| C | 控制面单元测试 | C++ 编译器 | OOM、EOS、Prefix/LRU 等不变量 |
| D | GPU 模型/采样行/Prefix 测试 | CUDA、GPT-2 checkpoint | 设备路径和独立参考一致 |
| E | 双卡 PD 测试 | 两张 GPU、checkpoint | KV 迁移、状态交接与正确性 |

本次对新增 A/B 实验和 GDB 操作做了实际运行验证，记录位于 `examples/`。
D/E 是既有验证入口，当前实现结果见 [任务 09—11 实测](../../benchmark/results/task09_11/README.md)。
文档扩写没有重新修改或扩展引擎功能。

## 2. 实验 A1：KV Cache 真的少算了吗

```bash
python doc/from_pytorch/examples/attention_and_pages.py cache
```

先预测：Prompt 长 5，生成 3 Token，完整前缀与缓存路径分别处理多少输入行？
再看实际检查：

- 完整 9 Token 前向与分成 3/2/4 的三次前向，全部 logits 一致到浮点误差范围内。
- 每层 K/V 一致，验证不止最后一个 Argmax。
- 修改最后一个 Token，不改变前 8 个位置的输出，验证因果性。
- 三步 Greedy 一致，前向输入行数由 18 降到 7。

读代码时从 [cache_lesson](examples/attention_and_pages.py) 到 `TinyCausalLM.forward`，再到 `generate`。
实验采用 float64 和随机小模型，最大误差可能随数学库实现略变；通过标准是 assert_close，
不是强求打印的小数完全一样。

可以尝试的练习：在你自己的副本中把 positions 改成每次从 0 开始，或把矩形 mask 改成
从左上角开始的 tril。先预测哪个对照会失败，再运行确认。不要据这个随机模型的输出质量
判断 KV Cache 是否正确。

## 3. 实验 A2：让物理页顺序与逻辑顺序不同

```bash
python doc/from_pytorch/examples/attention_and_pages.py pages
```

当前页表 `[3,0,4]`，P=4，长度 9。先在纸上列出 0..8 每个位置的物理页、页内偏移和 Slot。
脚本写入后重新按逻辑顺序读取，并比较完整 Attention 输出。

三个注意点：

1. Pool 中无效尾槽填 NaN，读取越过 context length 容易被发现。
2. 只验证页表相同不够，还要验证恢复的 K/V 数值和 Attention 结果。
3. Python 的 Gather 只是帮助理解；项目 CUDA 是在 Kernel 中直接查页。

读代码的位置是 `pages_lesson` 中两个 for/list comprehension，再去看
[cache_offset 与页读取](../../mini_vllm/cuda/paged_attention.cu)。

## 4. 实验 A3：把两请求压成一个输入矩阵

```bash
python doc/from_pytorch/examples/attention_and_pages.py packed
```

第 3 节例子固定输出：

```text
Position=[5,0,1,2]
Context=[6,1,2,3]
Slot=[5,16,17,18]
query_start=[0,1,4]
sample_rows=[0,3]
```

脚本还验证先 Gather hidden 再做 Linear，得到的行与完整 Linear 后再选同样行一致。
这就是任务 09 的数学基础。注意这里只比较独立的 Linear；真实模型先计算全部必要输入的
Transformer 与 KV，不能把 Prompt 的其他行在模型入口就删掉。

## 5. 实验 B：对真实 Scheduler 下断点

构建调试版，不改变正式引擎：

```bash
c++ -std=c++17 -O0 -g -gdwarf-4 -I. mini_vllm/demo.cpp -o /tmp/zyf_learning_demo
/tmp/zyf_learning_demo
```

`-gdwarf-4` 让调试信息兼容本机 GDB 9.2；运行语义与普通 debug 构建一致。
然后：

```bash
gdb -q /tmp/zyf_learning_demo
```

进入 GDB 后逐条执行：

```gdb
set pagination off
break mini_vllm/scheduler.hpp:89
run
print output.num_batched_tokens
print output.items[0].num_scheduled_tokens
print sampled_token_ids
list
```

本次源码第 89 行是 `Scheduler::commit` 函数体内的参数数量检查处。断在函数体内，更容易避免
在函数入口序言尚未建立参数位置时读取到无效值。源码改变后用下面命令重新定位：

```bash
rg -n 'sampled_token_ids.size\(\) != output.items.size' mini_vllm/scheduler.hpp
```

第一次断下时应该看到：

```text
output.num_batched_tokens = 8
output.items[0].num_scheduled_tokens = 6
sampled_token_ids = {1010, -1}
```

此时还没有 commit，因而 computed 尚未包含本轮结果。继续执行到 `mark_computed` 之后再
观察，才能验证状态从“计划执行”变成“已经执行”。参考
[GDB 实际输出](examples/gdb_verified_output.txt)。

`continue` 会在下一轮相同位置再停下。四轮样本数组应依次为：

```text
[1010,-1]
[1011,-1]
[1012,1020,1030]
[1021,1031]
```

如果你的运行环境禁止 ptrace，GDB 无法启动被调试进程。普通 demo 仍可直接运行并观察输出；
调试需要使用允许 ptrace 的环境。本次已在允许调试的进程环境中验证上述命令。

## 6. 实验 C：从行为追到测试不变量

```bash
make test_minivllm_control_plane
./test_minivllm_control_plane
```

源码：[test_mini_vllm_control_plane.cpp](../../dev/test_mini_vllm_control_plane.cpp)。按问题找函数：

| 你想确认什么 | 测试函数 | 重点看哪个 assert |
| --- | --- | --- |
| 页能否归还并复用 | test_block_allocation_release_and_reuse | free_blocks、Block Table |
| Chunk 未完成是否错误采样 | test_chunked_prefill | pending、样本 -1、是否完成 |
| OOM 是否分配半张请求页表 | test_oom_is_atomic_and_eos_releases_blocks | 失败前后请求页表、free count |
| 旧请求退出后新请求能否进入 | test_continuous_admission_and_retirement | running/waiting 与本轮 items |
| Prefix 命中是否跳过计算 | test_prefix_cache_hit_and_lru_eviction | computed=8、本轮 count=1 |

这里的 OOM 原子性针对请求新增页分配。开启 Prefix Cache 后，分配过程中可能驱逐缓存页，
不能把它解读成“失败时整个缓存索引也从未变化”。

## 7. 实验 D/E：掌握前三节后再运行

下面沿用已有可执行程序，不需要一次全跑：

```bash
make test_gpt2_cuda_sample_rows test_gpt2_cuda_prefix_cache test_gpt2_pd_engine GPU_COMPUTE_CAPABILITY=86
OMP_NUM_THREADS=8 ./test_gpt2_cuda_sample_rows
OMP_NUM_THREADS=8 ./test_gpt2_cuda_prefix_cache
OMP_NUM_THREADS=8 ./test_gpt2_pd_engine --cuda-graph
```

从 `test_gpt2_cuda_sample_rows.cu` 中固定 N=4、改变 R 的用例开始。理解同样输入行数为什么可能
没有任何需要采样的请求，再看 Prefix 的 18 Token 请求只调度 2 个 Token，最后看 PD 恢复
computed 和首 Token。

双卡测试需要两张设备可见，不能先设置 `CUDA_VISIBLE_DEVICES=0` 再运行。需要限制到这两张卡
时使用 `CUDA_VISIBLE_DEVICES=0,1`，它们在该进程内编号为 0、1。

## 8. 最小术语表

| 术语 | 本项目中如何理解 | 去哪里看 |
| --- | --- | --- |
| Token / Token ID | 模型词表中的离散输入编号，不一定等于一个汉字或单词 | Sequence::token_ids |
| Prompt / Completion | 已给定的输入 / 模型生成的输出 | num_prompt_tokens / num_completion_tokens |
| Logits | 词表的未归一化分数 | logits_matmul |
| Greedy / Sampling | 选最大分数 / 更一般的选择输出策略 | argmax_kernel；当前只实现 Greedy |
| Prefill / Decode | 处理 Prompt / 用历史 KV 处理后续新输入 | is_prefill |
| KV Cache | 各层历史 Key/Value 数值 | key_cache_ / value_cache_ |
| Prefix Cache | 跨请求复用相同完整前缀的 KV | apply_prefix_cache |
| Continuous Batching | 每轮可接纳/退出请求 | Scheduler::schedule / commit |
| Chunked Prefill | 将长 Prompt 分几轮计算 | try_schedule 的 count |
| Packed / Ragged | 压紧不同长度的有效输入，配边界/长度元数据 | prepare_packed_model_input |
| Position / Context Length | 当前绝对位置 / 该 Q 可见历史长度 | positions / context_lengths |
| Block Table / Slot Mapping | 历史逻辑页映射 / 本轮新 KV 写入槽位 | ModelInput |
| Worker / Runner | 设备执行单元 / 负责模型和输入执行的组件 | nano ModelRunner、项目 CUDA Runner |
| Preemption / Recompute | 暂停请求释放资源 / 恢复时重算历史 | 本地 nano preempt；本项目单卡未实现 |
| H2D / D2H | 主机到设备 / 设备到主机 | copy_metadata / 样本复制 |
| Pinned Memory | CUDA 可用于受控 DMA 传输的锁页主机内存 | copy_kv_to 中 cudaHostAlloc |
| Stream | 有序提交 GPU 工作的执行队列 | CudaStream |
| CUDA Graph | 可重放的一组 GPU 操作和依赖 | graph_key / cudaGraphLaunch |
| MHA / GQA / MQA | Q 与 KV 头数量不同的 Attention 组织方式 | GPT-2 为 MHA；nano Qwen 代码有 KV 头数 |
| TP / PP / DP / PD | 张量/层/请求/推理阶段的不同拆分 | 第 5 节对照表 |

术语只帮助定位职责，实际语义以项目实现为准。例如 Context 既可能指长度，也可能指 nano
中的元数据对象；Block 既可能指 KV 页，也可能指 CUDA 线程块，要结合类型和调用点区分。

## 9. 性能指标也从时间线推导

假设请求在 t=0 到达，输出 Token 的时间分别为 20、24、29、33 ms：

```text
到达 0 ───────── 首 Token 20 ─ 第二个 24 ─ 第三个 29 ─ 最后一个 33
```

| 指标 | 这个例子怎么计算 | 注意点 |
| --- | --- | --- |
| TTFT，首 Token 延迟 | 20 ms | 可包含排队、Prefill 等，具体看计时起点 |
| ITL，相邻 Token 延迟 | 4、5、4 ms | 是一个序列，能观察尾部慢点 |
| TPOT，首 Token 后平均时间 | `(33-20)/(4-1)=4.333 ms` | 不等于三个 ITL 的 P95 |
| 请求总延迟 | 33 ms | 包含首 Token 之前的等待 |
| 本窗口输出吞吐 | `4/0.033≈121.2 tok/s` | 不是只用 1/TPOT；多请求还需统一测量窗口 |

只有一个输出 Token 时，没有相邻输出区间；不要把分母 0 当成正常 TPOT。性能对比必须固定
工作负载、输出数、精度、预热和计时边界；修改口径后指标不可直接横比。

项目代码定位：[benchmark_gpt2_cuda_serving.cu](../../benchmark/benchmark_gpt2_cuda_serving.cu) 的
`run_once`、`percentile`，以及 Prefix Benchmark 的 `measure`。测量 GPU 执行必须在结束点
确认对应工作已完成，本项目 Runner 返回前同步 Stream。

## 10. 自测参考答案

### A. Prompt 5、生成 3，为什么输入行数是 18 与 7

完整前缀重算为 5+6+7；缓存为 5+1+1。最后一个输出不再输入模型，所以两种算法都不算
“Prompt 加全部三个输出”后的额外一轮。缓存路径仍会访问不断变长的历史 K/V，输入行数
减少不代表所有操作的开销都同倍数减少。

### B. 生成 Token 和计算它的 KV 有什么区别

生成 y0 是对 Prompt 最后位置 logits 做选择；计算 y0 的 KV，需要把 y0 再送过每层投影。
前者发生在 Prefill 末尾，后者发生在下一步 Decode。请求结束时最后一个输出无须再写 KV。

### C. 已有 3 个 Token，本轮两个新 Token，张量是什么形状

每层新 Q 为 `[H,2,D]`，拼接后的 K/V 为 `[H,5,D]`，分数 `[H,2,5]`。两个 Q 的绝对位置分别
是 3、4，可见长度分别是 4、5。对矩形矩阵随手使用左上角 tril 会漏掉应可见的历史。

### D. 同样的 Token ID 为什么不一定能共享 KV

各层 K/V 依赖对应位置的历史 hidden state，历史文本、位置、模型权重或其他影响前向的状态
不同，就可能不同。固定本项目模型时，以完整历史 Token 前缀确定共享身份；生产系统还可能
需要额外区分适配器等状态。

### E. 第二轮 B 执行 7 Token，为什么没有 7 个输出

输入额度不是输出额度。B Prompt 长 11，第一轮算 2、第二轮算 7，累计 9，还没完成全部输入，
所以返回 -1。本项目普通自回归生成一轮最多为每个完成输入的请求生成一个 Token。

### F. 一个请求已经 Running，为什么仍在 Prefill

Running 表示已接纳，Prefill 表示 computed 尚未到 Prompt 长度。长 Prompt 分块后在 Running
队列中持续推进。当前调度只保证先推进 Running，不保证所有 Decode 都严格先于 Prefill。

### G. A Decode 1，B Prompt 3 只算前 2，采样行是什么

Packed 行为 `[A0,B0,B1]`，A 完成本轮已有输入，B 尚未完成 Prompt，所以 `sample_rows=[0]`。
下一次 B 完成最后一个 Prompt Token 后才有它的采样行。

### H. 页表换成 [4,2,0]，P=4，位置 8 去哪里

逻辑页 2，页内偏移 0，物理页 table[2]=0，Slot=0。Slot=0 不代表绝对位置 0，位置仍是 8。

### I. Prefix Cache 与 CUDA Graph 可否共享同一个 Key

它们管理不同对象。相同形状、不同 Token 内容可以重放同一图，却不一定共享前缀；相同前缀
的请求也可能因为本轮输入/采样行数不同而使用不同图。

### J. PD 交接 Prompt 17、已输出 y0 后，D 应恢复什么

Prompt=17、computed=17、num_tokens=18、pending=1。迁移两页 Prompt KV，恢复 y0 作为下一步
输入。将 computed 设成 18 会跳过尚未计算的 y0。页号必须按 D 自己的分配映射，不能直接照搬 P。

## 11. 给自己留一张学习记录

每节都可以用下面四行记录，而不是只勾“读完”：

```text
我现在能解释的行为：
这个行为的生产者 → 消费者函数：
我实际运行的命令和关键输出：
我改了哪个输入，结果为何随之变化：
```

先完成第 1—3 节的四项记录，再读编译、nano 和 PD。这样每增加一个概念，都能放回已经
验证过的代码链路中。
