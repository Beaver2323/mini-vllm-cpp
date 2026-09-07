# llm.c 项目：简历、源码与面试学习手册

本手册基于当前工作区的 `train_gpt2.cpp`、`paged_kv_cache.hpp` 和 `train_gpt2.c`。
目标是能解释自己的实现、验证它、说明局限，并逐步补齐工程能力。
简历替换片段见同目录 `resume_project.tex`。以下区分现有能力与后续建设，不将计划写成已完成成果。

## 1. 当前代码能支撑什么

| 能力 | 代码依据 | 可表述的范围 |
| --- | --- | --- |
| 单 Token 增量解码 | `gpt2_forward_inference` | 只处理当前 Token，逐层复用历史 KV |
| 分页缓存池 | `KVCachePool` | CPU `malloc` 预分配 K/V 池，16 Token 一页 |
| 逻辑页映射 | `PageTable::block_tables` | 不同序列可映射到池内不连续的页 |
| 按需分配 | 生成循环中的取模判断 | 每到新页边界，从已预分配的池中领取页 |
| Attention 并行 | `omp parallel for collapse(2)` | batch 与 head 维度上的 CPU 并行 |
| 生命周期管理 | `KVCachePool` 析构 | 整个池销毁时释放，尚无单页归还接口 |
| 正确性验证 | `dev/test_paged_attention_resume.cpp`、`dev/test_gpt2_paged_inference.cpp` | 算子级独立参考及模型级全词表 logits 对齐 |

原简历中的“显存”“动态回收”“动态显存调度”超出了当前代码实现。
仓库存在 CUDA 训练代码，不代表这条新增的分页推理链路使用了 CUDA。
建议项目定位为“基于 llm.c 的 GPT-2 增量推理与分页 KV Cache”，技术栈写 C/C++、OpenMP。

## 2. 先分清两个优化

KV Cache 解决重复计算：自回归模型每步生成一个新 Token，旧 Token 在因果注意力下不受未来 Token 影响，因而各层旧 K/V 可以缓存。
下一步只算新 Token 的 Q/K/V、投影、MLP，再让新 Q 读取全部已有 K/V。
这里假设推理期间权重固定；当前代码每轮生成重新创建缓存池，避免跨训练更新继续使用旧 KV。

分页解决存储组织：连续 KV Cache 也能实现增量推理；分页使一个序列的缓存可以由多个不连续块组成。
不能把增量推理的计算收益全归因于分页。

只看单层 Attention、固定 batch/head/head_dim，长度为 t 时：

| 方式 | 单次生成步骤的 Attention 工作量 |
| --- | --- |
| 重算整个长度 t 的前缀 | O(t²) |
| 单 Token + 连续 KV Cache | O(t) |
| 单 Token + 分页 KV Cache | O(t)，额外包含页表寻址 |

这不是端到端延迟倍数。模型还有矩阵乘、词表投影、内存访存等成本。
原版生成入口实际上使用固定长度 T 的前向，因此对它做实验时应报告实际 T，不能直接把理论前缀复杂度当成测量结果。

## 3. 跟踪一个 Token

先读 `train_gpt2.cpp` 的生成循环，再读 `gpt2_forward_inference`，最后进入 `paged_attention_forward`。

每次循环：

1. 检查当前已缓存长度是否到达页边界，必要时领取页并填写页表。
2. 当前输入 Token 加上绝对位置为 `seq_len - 1` 的位置向量。
3. 每层完成 LayerNorm → QKV 投影 → 分页 Attention → 输出投影及残差 → LayerNorm → MLP 及残差。
4. 最终 LayerNorm、词表投影、Softmax，采样得到下一 Token。
5. 下一轮才为刚采样的 Token 计算并写入 KV。

为什么位置编码不能总取 0？传给基础算子的 T=1 表示本次只处理一个 Token，但它在整个上下文中仍然有自己的绝对位置。

为什么通常不缓存旧 Q？未来步骤用当前 Q 与历史 K/V 计算输出，不需要旧 Q。

当前实现从 EOT Token 开始生成；尚无独立的 prompt prefill API。
逐 Token 消费 prompt 可以作为正确性起点，批量 prefill 的高效实现属于后续工作。

## 4. 手算页表与地址

设页大小 P=16，序列 A 的 Block Table 是 `[5, 2, 9]`。

| 逻辑 Token 下标 | 逻辑页号 t / 16 | 页内偏移 t % 16 | 池内物理页号 |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 5 |
| 15 | 0 | 15 | 5 |
| 16 | 1 | 0 | 2 |
| 20 | 1 | 4 | 2 |
| 32 | 2 | 0 | 9 |

这里“物理页”是应用管理的缓存块，不是操作系统物理页帧，也不是实际 CPU 物理地址。
底层 K 和 V 分别通过一次连续 malloc 分配；不连续指一个序列拿到的块编号可以不连续。

缓存布局是 `[num_pages, num_layers, num_heads, page_size, head_size]`。
令 p 为物理页号、l 为层号、h 为头号、o 为页内 Token 偏移、d 为头内元素下标，则元素偏移为：

```text
offset = ((((p * L + l) * H + h) * P + o) * D + d)
address = cache_base + offset       // float 指针运算，单位是 float
byte_offset = offset * sizeof(float)
```

batch 没有显式出现在缓存维度里，因为不同 batch 序列通过各自页表领取不同的页。
当前“一页”包含一个序列这 16 个 Token 在所有层、所有头上的 K 或 V。

练习：令 L=2、H=2、D=4，求 t=20、l=1、h=0、d=3 的位置。
答案：p=2、o=4，offset=659，FP32 字节偏移=2636。

## 5. Attention 究竟算了什么

对当前 head，先把当前 K/V 写入它们的缓存位置，再遍历 s=0..t：

```text
score[s] = dot(q[t], k[s]) / sqrt(head_size)
weight[s] = exp(score[s] - max(score)) / sum(exp(score - max(score)))
out[t] = sum_s weight[s] * v[s]
```

访问 k[s] 和 v[s] 前需要通过页表找到实际地址。数学结果不应依赖物理页排列。
这也是正确性测试必须故意打乱页号的原因：只测顺序分配容易漏掉把逻辑页号误当物理页号的错误。
只遍历已存在的 Token，因此单 Token decode 不需要显式构造上三角 mask。

OpenMP 并行的是 `(b, h)`：各 worker 写不同的 head 区域和 scratch 区域，层循环仍是顺序执行。
这个 CPU 实现没有 CUDA thread block、共享内存或 warp reduction，也没有实现 FlashAttention。

## 6. 内存数字怎么算

缓存有效内容的字节数为 `2 × L × H × D × token_count × bytes_per_element`。
前面的 2 是 K 和 V；这里是普通多头注意力，H 为 KV 头数。

以 L=12、H=12、D=64、FP32 为例：

- 每 Token KV：`2 × 12 × 12 × 64 × 4 = 73,728 B = 72 KiB`。
- 每个 16 Token 页的 K+V：1.125 MiB。
- B=4、每序列容量 64 Token 时，整池容量为 18 MiB。

按需领取页不等于按需执行 malloc：当前池在构造时预留全部容量，后续只分配页的所有权。
每个非空序列最后一页最多浪费 15 个 Token 槽位。但本实现按最大生成长度创建足量页池，尚未通过多请求回收和复用展示内存利用率提升。
不能把这 18 MiB 当成进程总内存；权重、训练激活等仍占内存。

## 7. 已完成的验证与如何复现

从 `llm.c` 目录执行，使用用户指定的 zyf1 环境绝对路径：

```bash
conda run -p /home/miniconda3/envs/zyf1 g++ -std=c++17 -O2 -fopenmp dev/test_paged_attention_resume.cpp -o /tmp/zyf_paged_attention_test
OMP_NUM_THREADS=2 conda run -p /home/miniconda3/envs/zyf1 /tmp/zyf_paged_attention_test
```

本次结果：`max_abs_error=1.45372e-07`，退出码 0，阈值为 `1e-5`。
覆盖 B=2、L=2、NH=2、head_size=4，逐步长度 1..33，反向且交错分配的物理页。
参考实现直接从稠密 QKV 历史读取，采用 double 累加，不通过被测页表访问数据。
该算子测试不依赖权重文件，不代表性能达标。另一个模型级测试使用 GPT-2 124M 权重，
让两个异长请求在分页增量推理中动态加入和退出，活跃批次经历 1→2→1；请求 0 覆盖
长度 1..33，请求 1 覆盖长度 1..20。测试在每个有效位置比较完整前缀前向的 50,257 个
词表 logits；反向、交错分配物理页后最大绝对及相对误差均为 0。

## 8. 下一步怎样实质升级项目

按建议顺序推进，括号内为当前状态。

1. **独立推理入口与工作区（已完成）**：按 T=1 推理形状分配缓冲，并单独分配线性 Attention scratch。
2. **控制面接入模型（已完成）**：ModelRunner 已整理 token、position、context length、slot mapping 与 block table。
3. **调度与执行闭环（已完成）**：Engine 已连接 schedule、run、greedy sample、commit 和页回收。
4. **可信 Benchmark（已完成 CPU 基线）**：比较完整前缀重算、分页增量和连续批处理；固定权重、Token、线程数、编译选项及长度，记录预热后多次延迟、吞吐和内存口径。
5. **设备算子扩展（已完成 Multi-Token Prefill）**：FP32 CUDA PagedAttention 已接入 12 层 GPT-2 GPU ModelRunner；Packed Prefill 将同轮异长 Token 合并为 GEMM，通过完整词表 reference、memcheck、racecheck 与 RTX 3090 Benchmark。下一步实现低精度。

现有细节也值得修复：页表初始化为 0 会把未分配项伪装成合法页；softmax 最大值初值应使用负无穷而不是 -10000；裸指针所有权需要禁用拷贝或使用 RAII；推理入口缺少 Token 和上下文长度等输入校验。
`acts.preatt` 被用作 `[B, NH, max_seq_len]` scratch，但实际空间按训练 T 分配，需显式验证容量，不能依赖默认配置碰巧够大。

## 9. 一分钟项目讲述

“我基于 llm.c 的 GPT-2 实现了 CPU/CUDA 双路径增量推理和连续批处理。控制面包含 Sequence、BlockManager、Token Budget Scheduler 和 Chunked Prefill；ModelRunner 将调度结果转换为 Block Table、Slot Mapping 等设备元数据。我实现 FP32 CUDA PagedAttention，并接入完整 12 层 GPT-2，权重、KV Cache、中间激活和 logits 驻留 GPU，设备侧完成 Argmax。进一步将同轮异长 Prompt 压成 Packed Token Batch，使线性层由重复 GEMV 转为 GEMM，同时通过逐 Token Context Length 保证因果性。完整词表 logits 与 CPU Reference 最大绝对误差为 $2.67\times10^{-4}$，生成 Token 完全一致；固定负载达到 1838.148 tok/s，相对逐 Token GPU 基线提升 6.2 倍，Nsight 中 Kernel Launch 减少 87.6\%。”

理解每句话再用于面试。新增测试是在本次协作中补充的，应先读懂参考实现与测试覆盖范围。

## 10. 自测题

1. 若页表从 `[0,1,2]` 换成 `[5,2,9]`，输出应该变化吗？不应，前提是数据同步存放到相应页。
2. 为什么第 17 个 Token 要申请新页？它下标为 16，已有页只容纳下标 0..15。
3. PagedAttention 是否天然比连续 KV 更快？没有这种保证；它改变缓存组织，也增加地址间接访问。
4. 当前 continuous batching 闭环在哪里？`GPT2Engine::step` 依次调用 schedule、ModelRunner、greedy sample 和 commit，并在请求完成时释放 Block。
5. `free_pages` 是栈，是否意味着 CPU 调用栈或栈帧？不是，它是 vector 实现的索引栈，`num_free_pages` 表示有效空闲项数量。
6. 怎么证明速度收益来自哪里？用完整前缀重算→连续 KV 衡量缓存收益，再用连续 KV→分页 KV 分离布局与页管理影响。

阅读材料：[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)。论文解释分页 KV 的设计动机与服务系统应用；其中 vLLM 的吞吐数字不能用于本项目的简历。
