# 任务 11：双 GPU 的功能性 PD 分离

前置知识：[单请求生成的记账](from_pytorch/01_generation_and_kv.md)、[页地址](from_pytorch/03_pages_and_packed.md)、
[多卡拆分方式](from_pytorch/05_read_nanovllm_and_vllm.md#9-多卡概念先按拆什么区分)。
零基础先完成前三节；能手算首 Token 与 KV 的关系后，再进入本篇。

这一版已经在本机两张 RTX 3090 上执行真实 GPT-2 前向：GPU 0 负责 Prefill，GPU 1 负责
后续 Decode。两端各有完整模型副本和独立 KV Pool，使用 pinned host memory 中转 KV。

先掌握 Sequence 记账、分页寻址、Runner 和任务 09，再学习本篇。暂时不需要学习 TP、NCCL、
网络服务或分布式进程管理；本任务的完整边界是单机、同进程、双 GPU 的请求交接。

## 1. 先明确本机的通信路径

当前机器两卡拓扑为 PIX，同一 NUMA，未启用 NVLink；实测 `cudaDeviceCanAccessPeer` 双向
均为 0。存在 PCIe 通路，但本次环境不能直接使用 CUDA Peer Access 访问对端显存。

因此实现路径为：

```text
GPU 0 的 K/V 页 → D2H → portable pinned host buffer → H2D → GPU 1 的 K/V 页
```

这是实际数据复制，不是把源 GPU 指针传给目标 GPU。代码不依赖 NCCL 或 P2P，也没有把模型
权重切成两半。两卡各保留完整权重，新增显存占用是模型副本和各自缓存。

PD 是工作阶段的拆分；Tensor Parallel 是一个算子/模型权重的拆分。你现在可以讲前者的
功能实现，还不能称为张量并行实现。

## 2. 逐项代码地图

| 阅读顺序 | 需要理解 | 实现文件和搜索词 | 上层调用点 |
| --- | --- | --- | --- |
| 1 | 两个 Worker 的资源归属 | [gpt2_pd_engine.hpp](../mini_vllm/cuda/gpt2_pd_engine.hpp)，构造函数 | Benchmark/test 构造 `GPT2PDEngine` |
| 2 | 请求生命周期 | 同文件 `PDStage`、`PDRequest`、`add_request` | `run_pd` 调用 add_request |
| 3 | P/D 的一步调度 | 同文件 `step` | 外层 `while (!engine.is_finished())` |
| 4 | P 端首 Token 提交 | 同文件 `sequence.mark_computed`、`pending_` | P Runner 返回后 |
| 5 | 目标页分配和状态交接 | 同文件 `try_handoff` | 下一次 step 开头 |
| 6 | 页数据如何复制 | [gpt2_cuda_model_runner.cu](../mini_vllm/cuda/gpt2_cuda_model_runner.cu)，`Impl::copy_kv_to` | `try_handoff` |
| 7 | 设备切换 | 同文件 `DeviceGuard`、Runner 构造/析构/`run` | 所有涉及设备执行的 Runner 入口 |
| 8 | K/V 页的字节布局 | [paged_attention.cu](../mini_vllm/cuda/paged_attention.cu)，`cache_offset` | PagedAttention 读写缓存 |
| 9 | D 如何继续调度 | [scheduler.hpp](../mini_vllm/scheduler.hpp)，`try_schedule`、`commit` | PD step 中 `d_scheduler_` |
| 10 | 跨卡数值验证 | [test_gpt2_pd_engine.cu](../dev/cuda/test_gpt2_pd_engine.cu)，`test_remapped_transfer` | 测试 main |
| 11 | 页边界/背压/EOS 验证 | 同文件 `test_pipeline` | 测试 main，普通与受限页池两次 |
| 12 | 观察真实执行轨迹 | [benchmark_gpt2_pd_serving.cu](../benchmark/benchmark_gpt2_pd_serving.cu)，`run_pd` | Benchmark main |

用函数名定位：

```bash
rg -n 'try_handoff|mark_computed|std::async|ensure_capacity|pending_' \
  mini_vllm/cuda/gpt2_pd_engine.hpp
rg -n 'DeviceGuard|copy_kv_to|cudaHostAlloc|page_bytes' \
  mini_vllm/cuda/gpt2_cuda_model_runner.cu
```

## 3. 整体调用关系

```text
benchmark: run_pd
  └─ GPT2PDEngine::add_request → waiting_
  └─ while 未完成：step
       ├─ try_handoff：上轮 P 完成的请求尝试进入 D
       │    ├─ D 端分配自己的物理页
       │    ├─ p_runner.copy_kv_to(d_runner, source, target, computed)
       │    ├─ 恢复 target 的 computed 和首 Token
       │    ├─ d_scheduler.add(target)
       │    └─ 释放 P 端页，更新稳定 Request Handle
       ├─ 构造 P 本轮 Chunk；d_scheduler.schedule
       ├─ p_runner.run(P) 与 d_runner.run(D)
       ├─ 提交 D：沿用 Scheduler::commit
       └─ 提交 P：未完成继续 Prefill；完成则排队迁移或直接结束
```

本版 P 同时处理一个 Prompt，可以切 Chunk；D 使用已有 Scheduler 对多个请求做 Continuous
Batching。这样复用已有控制面，只增加阶段分工和交接状态，便于逐步学习。

## 4. 最容易出错的地方：首 Token 和 KV 的关系

假设 Prompt 有 17 个 Token，要求生成 4 个新 Token。P 一次完整 Prefill 后：

| 时刻 | num_prompt_tokens | num_computed_tokens | num_tokens | pending_tokens |
| --- | ---: | ---: | ---: | ---: |
| 初始 | 17 | 0 | 17 | 17 |
| P 完成 Prompt 计算 | 17 | 17 | 17 | 0 |
| P 追加首 Token `y0` | 17 | 17 | 18 | 1 |
| D 恢复完成 | 17 | 17 | 18 | 1 |
| D 输入 y0 并生成 y1 | 17 | 18 | 19 | 1 |

P 有 Prompt 17 个 Token 的 KV，但没有 y0 的 KV。y0 是从 Prompt 最后位置的 logits 采样得到，
还没有作为模型输入。D 的第一次前向必须输入 y0，Position=17，Context Length=18。

交接中的真实代码：

```cpp
target->mark_computed(source.num_computed_tokens());
target->append_token(source.token_ids().back());
d_scheduler_.add(target);
```

`target` 起初由原 Prompt 构造，状态仍是 Waiting。恢复 computed 后再 append 首 Token，满足
Sequence 的“只有全部已有输入计算完成后才能 append”约束。下一次 D Scheduler 调用
`is_prefill()` 得到 false，自动进入 Decode。

如果误把 computed 设成 18，就会跳过 y0 的前向；如果只恢复 Prompt 不追加 y0，就没有待计算
输入。这两种错误都会破坏生成链路。

## 5. Request Handle 为什么与 Sequence 分开

用户得到 `shared_ptr<PDRequest>`，其中 `request->sequence` 当前指向所在 Worker 的状态对象。
交接时构造新的 D Sequence，原 P Sequence 的页表由 P BlockManager 回收，然后替换指针：

```cpp
const auto request_id = source.request_id();
p_blocks_.release(source);
pending_->sequence = std::move(target);
pending_->stage = PDStage::Decoding;
```

外部保存 PDRequest 就能始终访问当前状态。不要跨交接长期保存旧 `Sequence&`；它的所有权会
改变，旧对象可能被销毁。这里先保存 request_id，也是为了避免替换后继续访问失效引用。

阶段变化为：

```text
WaitingPrefill → Prefilling → TransferPending → Decoding → Finished
                         └─ 首 Token 达到 max_new_tokens/EOS → Finished
```

## 6. 逻辑页号与两端物理页号

两端 BlockManager 是独立的，不能把 P 的页表直接交给 D 使用。例如：

```text
逻辑页号            0       1
P block_table      [0,      1]
D block_table      [1,      2]
需要复制            P0→D1   P1→D2
```

`copy_kv_to` 按逻辑页序号 i 选择两端各自的物理页地址：

```cpp
source_cache + source.block_table()[i] * page_bytes
host_staging + (kv * num_pages + i) * page_bytes
target_cache + target.block_table()[i] * page_bytes
```

这是对应实际拷贝的地址表达式简写，完整调用见 `cudaMemcpyAsync` 两组循环。`kv=0` 对应 K，
`kv=1` 对应 V。不能假设两张卡碰巧分到相同物理页号就省略映射。

底层每种缓存布局为：

```text
[physical_page, layer, head, page_offset, head_dimension]
```

一个物理页包含该页 16 个 Token 在全部层、全部注意力头的 K（或 V）。因此每种缓存中单页
连续存放，可直接复制整页。GPT-2 124M 的 FP16 数据量：

```text
单 Token 的 K+V = 2 × 12层 × 12头 × 64维 × 2字节 = 36 KiB
单页的 K+V      = 16 × 36 KiB = 576 KiB
17 Token Prompt = ceil(17/16) × 576 KiB = 1.125 MiB
```

最后一页只用一个槽位时仍复制整页。未使用槽位可能保留旧值，但 Attention 只读取
`context_length` 以内的位置；D 每次先写新 Token KV，再读到当前合法长度，不使用这些尾部值。

## 7. Host staging 的生命周期与同步

实现先校验设备、dtype、层数、头数、通道、Token 前缀和页表容量/范围，再执行：

```cpp
cudaHostAlloc(&staging, result.payload_bytes, cudaHostAllocPortable);
// P stream: 每页 K/V 做 D2H
cudaStreamSynchronize(p_stream);
// D stream: 按目标页号逐页做 H2D
cudaStreamSynchronize(d_stream);
cudaFreeHost(staging);
```

这是调用顺序简写，真实 Stream 取自两端 Runner。`Portable` 使 pinned buffer 可用于两端 CUDA
上下文。D2H 完成后才能让 D 使用主机内容，H2D 完成后才能释放主机缓冲区。

同理，源页在交接成功前必须继续由 P 持有。否则下一请求可能重新写入源页，而迁移还未完成。
本版同步交接让这个生命周期容易检查。

`KVTransferStats::payload_bytes` 是 K+V 的有效搬运页载荷，包含尾页 padding。主机链路总传输量
是它的两倍，因为 D2H/H2D 各一次。`total_ms` 包括 pinned 分配、设备切换、两段复制/同步和释放，
不同于纯 memcpy 带宽测试。每次分配 staging 简单但较贵，复用/流水化属于后续性能工作。

底层接口要求两个 Runner 使用相同权重、目标页独占、调用期间没有并发 run。Engine 从同一份
checkpoint 构造双副本，并在两个前向完成后交接，满足这些前提；接口没有计算权重哈希。

## 8. 为什么 CUDA Runner 要固定设备

CUDA 当前设备是线程局部状态。两个 Runner 构造、执行和析构都必须落在自己的设备上：

```cpp
DeviceGuard guard(device_id_);
return impl_->run(output);
```

Guard 先记住调用线程原设备，再切到 Runner 所属设备，离开时恢复原设备。构造时也先设置设备，
让权重、缓存、Stream 和 cuBLAS Handle 分配在正确的 GPU。析构显式切换设备后销毁资源。

否则容易出现“第二个 Runner 改了当前设备，之后第一个 Runner 在错误的上下文操作 Stream”
的问题。调用者无需在每一步手动 cudaSetDevice。

## 9. 两端如何同时推进

当同一轮有 P 和 D 工作时：

```cpp
auto prefill = std::async(std::launch::async, [&] {
    return p_runner_.run(p_output);
});
d_samples = d_runner_.run(d_output);
p_samples = prefill.get();
```

P 在独立主机线程中向 GPU 0 的 Stream 提交，当前线程向 GPU 1 提交 D。两个 Runner 独立持有
输入/输出 Buffer、图缓存和 cuBLAS Handle，互不共享可写设备内存。

`get()` 之后两端工作均已完成，再提交 CPU 状态。KV 交接仍在 step 开头同步完成，因此它没有
与计算重叠。本版 `concurrent_submissions` 计数只表示两端同时提交的轮数；实际 GPU Kernel
时间重叠需要看 Nsight 时间线，不能只根据线程数得出性能结论。
本机 node 级采集已观察到真实重叠，但累计量较小，详见结果文档的 Nsight 小节。

## 10. D 忙时的背压与容量策略

`try_handoff` 先检查 D 最大请求数，再尝试分配目标页。资源不足时保留 pending，不释放源页。
D 继续运行已接纳请求，完成后回收页，后续 step 再尝试交接。

本版最多保留一个待迁移请求；存在 pending 时，P 暂停接纳新请求。它形成有界背压，避免 P
无限积压已经算完却无处存放的 KV。

D 为每个接纳请求预留 `prompt + max_new_tokens - 1` 覆盖的全部页：

```cpp
const auto reserve_tokens = source.num_prompt_tokens() +
    pending_->sampling.max_new_tokens - 1;
if (!d_blocks_.ensure_capacity(*target, reserve_tokens)) return;
```

这样牺牲一部分缓存利用率，换来无需抢占的确定进度，避免多个请求各占部分页后一起无法扩容。
P 也在接纳时为整个 Prompt 分页预留。实际 GPU 计算仍按 Chunk 推进，实际迁移只复制已计算页。

单请求即超过某端页池的输入在 add_request 时拒绝；迁移/接纳抛异常时回收刚分配的 D 页，保留
P 状态。CUDA 严重设备错误、进程崩溃后的恢复不在本版能力内。

## 11. 怎样运行和观察

```bash
conda activate zyf1
make test_gpt2_pd_engine benchmark_gpt2_pd_serving GPU_COMPUTE_CAPABILITY=86
OMP_NUM_THREADS=8 ./test_gpt2_pd_engine
OMP_NUM_THREADS=8 ./test_gpt2_pd_engine --cuda-graph
OMP_NUM_THREADS=8 ./benchmark_gpt2_pd_serving
```

Benchmark 的预热轨迹会打印每轮 P Token 数、D Token 数、交接次数、并发提交标记，以及每个
请求的 KV 载荷、交接时间和输出 Token ID。打印位于计时结束后。

典型轨迹开头：

```text
step=0 GPU0_prefill=17 GPU1_decode=0 handoff=0 concurrent=0
step=1 GPU0_prefill=32 GPU1_decode=1 handoff=1 concurrent=1
step=2 GPU0_prefill=1  GPU1_decode=1 handoff=0 concurrent=1
step=3 GPU0_prefill=32 GPU1_decode=2 handoff=1 concurrent=1
```

第二轮开始，D 处理 A 的首个输出 Token，P 处理 B 的 Prompt；B Prompt 完成后也加入 D。
输出暂为 Token ID，模型使用本地 `gpt2_124M.bin`，没有加入分词器或 HTTP 接口。

Benchmark 测 17/33/49 Token 的三个 Prompt，各生成 8 Token；单卡使用最大 3 请求、Token
Budget 32，PD 的 P Budget 32、D 最大 3 请求；都使用 FP16、关闭 Fusion、开启采样行裁剪。
加载和一次预热不计时，各模式重复 7 次，交替先后顺序，逐请求输出对齐。

详细耗时见 [原始结果与口径](../benchmark/results/task09_11/README.md)。单卡对照能回答本例
功能拆分带来的成本，不能代替同样两张 GPU、各跑完整请求的数据并行基线。

## 12. 测试怎样覆盖关键错误

| 测试点 | 函数 | 检查内容 |
| --- | --- | --- |
| 故意使用不同物理页号 | `test_remapped_transfer` | D 先占一个 Block，再迁移 P 的两页 |
| 迁移后完整词表一致 | 同上 | 连续三步 Decode，比两端 50,257 个 logits，误差 < 1e-5 |
| 错误交接拒绝 | 同上 | 相同 Runner、computed 越界、Token 前缀不同 |
| 页边界和输出一致 | `test_pipeline` | Prompt 1/16/17/31/32/33；与单卡和 CPU Greedy 对齐 |
| D 容量受限 | `constrained=true` | D 只有 3 页，验证排队后继续推进，无死锁 |
| 首 Token 即结束 | max_new_tokens=1 与 EOS 分支 | 不迁移 KV，P 直接释放 |
| 全部资源回收 | 流水结束的 assert | 两端 free_blocks 恢复初始值 |
| CUDA Graph | `--cuda-graph` | 两个 Runner 各自缓存和重放自己的图 |

## 13. 学习练习和当前边界

1. 不看代码，写出 17 Token Prompt 交接前后五个 Token 计数；解释为何复制两页但只恢复 computed=17。
2. 手画 P 页表 `[4,0]`、D 页表 `[2,5]` 的复制地址；定位代码里对应的 `i` 循环。
3. 把 Decode Pool 改成只能放一个请求，观察 pending 为什么阻止 P 无限接纳。
4. 找到源页释放的唯一交接位置，解释为何它在 D 的 H2D 同步之后。
5. 比较 `transfer_ms` 和总时间，说明复用 pinned buffer、异步搬运可能针对哪部分开销。

当前边界：同进程双 GPU、Greedy、同一 GPT-2 权重、手工指定两个设备、同步 KV 交接、P 一次
处理一个 Prompt、D 多请求连续批处理。PD 路径未组合 Prefix Cache；没有 TP、多机 RDMA、
网络服务、动态 Worker 比例、抢占和容错。完成请求的管理记录保留到 Engine 销毁，适合有限
请求的学习实验，长期服务还需设计请求记录清理策略。

简历可据实写：“实现同机双 GPU Prefill/Decode 分离，完成独立页池的 KV 迁移、逻辑到物理
页重映射、请求状态交接和背压；支持 P/D 并发提交，通过跨卡 logits、CPU/单卡输出及页回收验证。”
速度提升需要对应实测证据，本版不据此声称生产级分布式推理或多卡加速。
