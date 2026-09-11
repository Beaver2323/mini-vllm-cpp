# 任务 11：双 GPU 的功能性 PD 分离

PD 分离是把 Prompt 的 Prefill 与后续 Decode 交给不同执行端。本项目用同一进程、两张 GPU、两份完整 GPT-2 权重和两套独立 KV 页池实现功能性分离。

学习目标：跟踪一个 17 Token 请求跨卡续写，证明迁移后继续生成与原卡一致，并能解释背压、资源所有权和当前性能边界。

## 1. 先与熟悉的多卡 PyTorch 概念区分

| 方式 | 本质分工 | 本项目这一任务是否实现 |
| --- | --- | --- |
| 数据并行 / 多副本 | 不同卡各自完成不同请求的全过程 | 不是当前 PD 执行方式，也未提供相应双副本性能基线 |
| Tensor Parallel | 一个算子的权重与计算分片到多卡 | 未实现 |
| 按模型层切分的 Pipeline Parallel | 不同卡持有不同层 | 未实现 |
| PD 分离 | 同一请求的 Prefill 与 Decode 分到不同执行端 | 已实现功能路径 |

P 卡与 D 卡都有完整模型，不是把一份大模型拆开放到两张卡。两卡模型权重占用合计增加，也不能借此宣称单模型容量扩展到两卡显存之和。

当前交接经过 pinned host 内存中转。文末保留本机器拓扑和性能记录，这不是 NCCL AllReduce、RDMA 或跨机器 KV 服务。

## 2. 先打开这四个文件

| 阅读顺序 | 文件 | 要回答的问题 |
| --- | --- | --- |
| 1 | [gpt2_pd_engine.hpp](../mini_vllm/cuda/gpt2_pd_engine.hpp) | 状态机和计算顺序 |
| 2 | [copy_kv_to](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L836) | 数值怎样按逻辑页迁移 |
| 3 | [test_gpt2_pd_engine.cu](../dev/cuda/test_gpt2_pd_engine.cu) | 怎样证明目标页号不相同也正确 |
| 4 | [benchmark_gpt2_pd_serving.cu](../benchmark/benchmark_gpt2_pd_serving.cu) | 性能比较包含哪些工作 |

```text
用户持有 PDRequest handle
  → waiting
  → P Runner：分块 Prefill
  → P 采样首 Token
  → pending 等待 D 可接纳
  → 分配 D 页 → 迁移 KV → 恢复 D Sequence
  → D Scheduler / Runner：逐轮 Decode
  → Finished → 回收页
```

这里“用户持有 handle”很关键：交接时内部 Sequence 对象会替换，外部应通过稳定的 PDRequest 读取当前状态。

## 3. 状态机比 CUDA 拷贝更早需要设计

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 13—23 行](../mini_vllm/cuda/gpt2_pd_engine.hpp#L13)。以下为当前文件的原样摘录。

```cpp
enum class PDStage { WaitingPrefill, Prefilling, TransferPending, Decoding, Finished };

// handle 保持稳定；交接时 Sequence 切换为 D 端对象，不能长期保存旧 Sequence 引用。
struct PDRequest {
    std::shared_ptr<Sequence> sequence;
    SamplingParams sampling;
    PDStage stage = PDStage::WaitingPrefill;
    KVTransferStats transfer;
    std::vector<int> source_pages;
    std::vector<int> destination_pages;
};
```

| PDStage | 此时谁持有主要状态 | 可以做什么 |
| --- | --- | --- |
| WaitingPrefill | waiting 队列中的请求 | 等待 P 接纳 |
| Prefilling | P Sequence 与 P 页 | 继续计算 Prompt chunk |
| TransferPending | P Sequence 与已计算 KV | 等待 D 空间；不能先释放源页 |
| Decoding | D Sequence 与 D 页 | 继续生成后续 Token |
| Finished | 请求结果 handle | 返回完整输出；KV 页已回收 |

不要混淆 `PDStage` 与 `SequenceStatus`。前者表达请求位于流水线哪一段，后者用于单端调度里的 Waiting/Running/Finished。

`source_pages/destination_pages` 保存交接时的页号快照，用于验证和解释迁移。请求完成后内部页表会清空，快照才让你仍能看到当时发生了什么。

## 4. 17 Token Prompt：首 Token 的 KV 为什么还没有算

设 Prompt 为 `p0...p16`，目标生成 4 个输出 `g0...g3`：

| 时刻 | 已知 Token 数 | computed | 哪些 KV 有效 | 下一步输入 |
| --- | ---: | ---: | --- | --- |
| 刚提交 | 17 | 0 | 无 | Prompt |
| P 完成 Prefill | 17 | 17 | `p0...p16` | 用末行 logits 采样 |
| P 追加 g0 | 18 | 17 | 仍只有 Prompt KV | g0 |
| D 接管 | 18 | 17 | 迁移来的 Prompt KV | g0 |
| D 第 1 轮后 | 19 | 18 | Prompt + g0 | g1 |
| D 第 2 轮后 | 20 | 19 | Prompt + g0 + g1 | g2 |
| D 第 3 轮后 | 21 | 20 | Prompt + g0 + g1 + g2 | 结束，g3 不再输入 |

最容易出错的是把 D 的 computed 设置成 18。g0 是预测出来的 ID，并没有送进模型计算；设成 18 会跳过 g0 的前向，让后续状态不完整。

同样，不能在 D 端重算整个 Prompt 作为“迁移后继续执行”的实现，否则没有真正复用迁移 KV，也无法检验交接内容是否正确。

## 5. P 端为何只手动提交 Prompt 这一段

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 133—150 行](../mini_vllm/cuda/gpt2_pd_engine.hpp#L133)。以下为当前文件的原样摘录。

```cpp
        if (!p_output.items.empty()) {
            auto& sequence = *prefilling_->sequence;
            sequence.mark_computed(p_output.num_batched_tokens);
            if (sequence.pending_tokens() == 0) {
                sequence.append_token(p_samples[0]);
                record_sample(result, sequence.request_id(), p_samples[0]);
                if (sequence.should_finish_after(p_samples[0])) {
                    sequence.set_status(SequenceStatus::Finished);
                    prefilling_->stage = PDStage::Finished;
                    p_blocks_.release(sequence); // 首 Token 即结束：无需迁移 KV。
                } else {
                    prefilling_->stage = PDStage::TransferPending;
                    pending_ = prefilling_; // P 页在交接完成之前仍被持有。
                }
                prefilling_.reset();
            } else if (p_samples[0] != -1) {
                throw std::logic_error("partial PD prefill produced a sample");
            }
```

P 端每次 mark 本轮计算数；只有 Prompt 全部完成才追加第一个输出。之后分两种情况：

- 首 Token 已满足输出数或 EOS：直接完成，释放 P 页，不迁移。
- 还需继续生成：进入 pending，保留 P 页，等 D 接管。

P 端不把请求放回普通单卡 Scheduler 继续 Decode，职责在这个分支明确切断。D 端则复用现有 Scheduler 的 normal commit，避免重复实现全部 Decode 状态逻辑。

部分 Prompt chunk 必须返回 `-1`。若中间 chunk 提前产生对外采样，后续输入就会被错误改写。

## 6. D 为什么一次预留整个生成上限

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 169—177 行](../mini_vllm/cuda/gpt2_pd_engine.hpp#L169)。以下为当前文件的原样摘录。

```cpp
    void try_handoff(PDStepResult& result) {
        if (!pending_ || d_scheduler_.num_waiting() + d_scheduler_.num_running() >= max_decode_)
            return;
        auto& source = *pending_->sequence;
        std::vector<int> prompt(source.token_ids().begin(),
                                source.token_ids().begin() + source.num_prompt_tokens());
        auto target = std::make_shared<Sequence>(source.request_id(), std::move(prompt), pending_->sampling);
        const auto reserve_tokens = source.num_prompt_tokens() + pending_->sampling.max_new_tokens - 1;
        if (!d_blocks_.ensure_capacity(*target, reserve_tokens)) return;
```

`reserve_tokens = prompt_len + max_new_tokens − 1`，与任务 01 的最大实际输入量一致。D 在接纳时就为该请求预留最大所需页，后续 Decode 不再因临时扩页与其他请求相互等待。

教学例子：D 只有 2 页，两个请求各当前需要 1 页，但继续生成后各要 2 页。如果只分当前 1 页就同时接纳，两者最终可能都等待新的空闲页，而当前系统没有抢占恢复机制。

预留策略会降低同时接纳的请求数量，并可能为提前 EOS 的请求保留暂时用不到的页；它换取的是更容易验证的完成能力。这个版本优先让状态机可理解，没有实现复杂动态超售或换出。

在 `add_request` 阶段还会提前拒绝“单个请求自己就放不进某端池子”的输入，否则它可能永远留在等待状态。

## 7. 交接的完整提交点

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 178—197 行](../mini_vllm/cuda/gpt2_pd_engine.hpp#L178)。以下为当前文件的原样摘录。

```cpp
        try {
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
        pending_.reset();
    }
```

顺序必须完整理解：

1. 目标页先分配好。
2. 把源已计算 KV 迁到目标页，等待迁移完成。
3. 目标 computed 恢复为 Prompt 长度，再追加 P 已生成的首 Token。
4. D Scheduler 接纳目标 Sequence。
5. 释放 P 页，把稳定 handle 的 Sequence 指针切换到 D 对象。
6. 记录已交接请求，清空 pending。

如果第 2—4 步发生异常，catch 归还目标页，并保留源状态，再向上传播异常。这不是完整服务故障恢复：CUDA 执行错误后能否继续使用 Runner 没有保证，也没有自动重试策略。

`request_id` 在替换 Sequence 前保存，避免后续继续访问旧对象引用。外部代码也不应长期缓存 `auto& old = *handle->sequence` 后跨交接使用；每次通过 handle 读取当前 Sequence。

## 8. copy_kv_to 校验了什么，没自动验证什么

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 836—852 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L836)。以下为当前文件的原样摘录。

```cpp
    KVTransferStats copy_kv_to(Impl& destination, const Sequence& source,
                              const Sequence& target, std::size_t computed_tokens) {
        if (this == &destination || config_.device_id == destination.config_.device_id ||
            config_.data_type != destination.config_.data_type ||
            config_.num_layers != destination.config_.num_layers ||
            config_.num_heads != destination.config_.num_heads ||
            config_.channels != destination.config_.channels ||
            computed_tokens == 0 || computed_tokens > source.num_computed_tokens() ||
            computed_tokens > target.num_tokens() ||
            computed_tokens > static_cast<std::size_t>(destination.max_context_length_)) {
            throw std::invalid_argument("incompatible KV handoff");
        }
        if (!std::equal(source.token_ids().begin(),
                        source.token_ids().begin() + computed_tokens,
                        target.token_ids().begin())) {
            throw std::invalid_argument("KV handoff token prefix mismatch");
        }
```

函数拒绝同 Runner/同设备、dtype 与模型主要维度不兼容、迁移长度越界、Token 前缀不同等情况。接下来还会在任何拷贝前检查源和目标页表长度与页号合法性。

但“维度相同”不代表“模型权重相同”。当前 PD Engine 从同一份 host 参数构造两端来满足这个前提，拷贝函数没有逐字节比较或哈希全部权重。

调用者也要保证目标页可独占写入，源/目标 Runner 没有正在并发修改这些 KV。`copy_kv_to` 并不是可接受任意外部页表的安全共享内存服务接口。

区分运行时检查与上层保证，是阅读系统代码比阅读单个 Tensor 算子多出来的重要工作。

## 9. 迁移按逻辑页对应，不能照抄物理页号

假设 Prompt 17 个 Token，需要 2 页，源页表 `[0,1]`，目标因已有请求占用而得到 `[1,2]`：

```text
逻辑第 0 页：GPU0 physical 0 → host K0/V0 → GPU1 physical 1
逻辑第 1 页：GPU0 physical 1 → host K1/V1 → GPU1 physical 2
```

GPU0 的物理页 1 与 GPU1 的物理页 1 没有全局身份关系。它们属于独立页池，含义由各自 BlockManager 决定。

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 866—890 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L866)。以下为当前文件的原样摘录。

```cpp
        const std::size_t page_bytes = key_cache_.bytes() / num_pages_;
        result.payload_bytes = 2 * result.num_pages * page_bytes;
        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const auto start = Clock::now();
        void* staging = nullptr;
        check_cuda(cudaHostAlloc(&staging, result.payload_bytes, cudaHostAllocPortable),
                   "allocate pinned KV staging");
        try {
            const auto begin_d2h = Clock::now();
            {
                DeviceGuard guard(config_.device_id);
                for (int kv = 0; kv < 2; ++kv) {
                    const auto* cache = static_cast<const char*>(
                        kv == 0 ? key_cache_.data() : value_cache_.data());
                    for (std::size_t i = 0; i < result.num_pages; ++i) {
                        check_cuda(cudaMemcpyAsync(
                            static_cast<char*>(staging) + (kv * result.num_pages + i) * page_bytes,
                            cache + source.block_table()[i] * page_bytes, page_bytes,
                            cudaMemcpyDeviceToHost, stream_.get()), "stage source KV page");
                    }
                }
                check_cuda(cudaStreamSynchronize(stream_.get()), "complete KV D2H");
```

`page_bytes` 是单个 K 物理页包含全部层、全部头、16 个位置的字节数。循环先按 K/V 两份缓存，再按逻辑页 i，把源物理页放进连续 staging。

staging 布局为：

```text
[K logical_page0][K logical_page1]...[V logical_page0][V logical_page1]...
```

源端 D2H 同步后，目标端才开始读取 host 内容：

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 894—906 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L894)。以下为当前文件的原样摘录。

```cpp
                DeviceGuard guard(destination.config_.device_id);
                for (int kv = 0; kv < 2; ++kv) {
                    auto* cache = static_cast<char*>(kv == 0
                        ? destination.key_cache_.data() : destination.value_cache_.data());
                    for (std::size_t i = 0; i < result.num_pages; ++i) {
                        check_cuda(cudaMemcpyAsync(
                            cache + target.block_table()[i] * page_bytes,
                            static_cast<char*>(staging) + (kv * result.num_pages + i) * page_bytes,
                            page_bytes, cudaMemcpyHostToDevice, destination.stream_.get()),
                            "restore destination KV page");
                    }
                }
                check_cuda(cudaStreamSynchronize(destination.stream_.get()), "complete KV H2D");
```

这一步通过目标自己的 `block_table[i]` 选地址，实现页号重映射。直接用 source 页号作为目标地址会在目标分配不同页时读写错误请求的空间。

## 10. 为什么尾页传满，但只允许读有效部分

Prompt=17 时，第二页只有第 0 个位置有效。当前为了简单按完整物理页传输，所以也会搬运尾页其余位置的内容。

正确性来自目标 context 仍是 17，并且下一次 Decode 在 position=17 写入 g0 的 K/V 后，才以 context=18 读取。无效尾部不能因为“已经传过去”就被视为有效历史。

传满页有额外带宽成本，但减少了跨层跨头分散打包的复杂性。当前没有实现只传有效尾部或压缩 KV 的路径。

按 GPT-2、FP16 计算：

```text
单页 K+V = 2 * 12 * 16 * 768 * 2 = 589,824 字节
17 Token → 2 页 → payload = 1,179,648 字节 = 1.125 MiB
经 host：D2H 搬一次，H2D 再搬一次
总方向传输字节 = 2 * payload = 2.25 MiB
```

`payload_bytes` 表达逻辑迁移数据量，不是两段链路字节数之和。报告有效传输带宽时要写清分子采用哪一种口径。

## 11. 设备上下文为什么每个线程都要处理

源码：[mini_vllm/cuda/gpt2_cuda_model_runner.cu，第 50—60 行](../mini_vllm/cuda/gpt2_cuda_model_runner.cu#L50)。以下为当前文件的原样摘录。

```cpp
// CUDA 当前设备是线程局部状态；每次入口设置并恢复，避免跨卡析构/执行。
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        check_cuda(cudaGetDevice(&previous_), "get current device");
        if (device != previous_) check_cuda(cudaSetDevice(device), "set runner device");
    }
    ~DeviceGuard() { cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};
```

CUDA 当前设备是线程局部状态。主线程先 `cudaSetDevice(0)`，不等于之后启动的 async 线程就自动在正确设备上执行所有操作。

每个 Runner 绑定自己的 device ID，入口设置目标设备，退出恢复之前设备。跨卡拷贝也分别在 source/destination 的 guard 下使用对应 stream。

析构同样要在资源所属设备的上下文中完成。多卡代码中只检查 forward 的 device，不检查销毁路径，是常见的资源错误来源。

## 12. P(B) 与 D(A) 怎样并发推进

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 113—123 行](../mini_vllm/cuda/gpt2_pd_engine.hpp#L113)。以下为当前文件的原样摘录。

```cpp
        std::vector<int> p_samples, d_samples;
        if (!p_output.items.empty() && !d_output.items.empty()) {
            result.concurrent_submissions = true;
            auto prefill = std::async(std::launch::async, [&] { return p_runner_.run(p_output); });
            d_samples = d_runner_.run(d_output);
            p_samples = prefill.get(); // 两端均完成后才提交 CPU 状态和发起下一次交接。
        } else if (!p_output.items.empty()) {
            p_samples = p_runner_.run(p_output);
        } else {
            d_samples = d_runner_.run(d_output);
        }
```

主线程执行 D，另一个 host 线程提交 P；两端各有自己的 Runner 与 stream。`prefill.get()` 等 P 完成，之后才提交 CPU 状态和进行下一次交接。

```text
一轮示意：
GPU P： [新请求 B 的 Prompt chunk]
GPU D： [老请求 A 的 Decode]
CPU：   等两端完成 → commit → 下一轮开始时尝试交接
```

`concurrent_submissions=true` 证明代码走了双端提交分支，但不能证明 GPU kernel 实际重叠了多少。GPU timeline 才能提供执行时间上的重叠证据。

当前 KV 迁移仍是同步阶段，没有与其他计算重叠的传输流水。计算可并发、传输可并发、端到端变快，是三个需要分别验证的结论。

## 13. 背压：D 满时为何暂停接纳新 P 请求

源码：[mini_vllm/cuda/gpt2_pd_engine.hpp，第 89—100 行](../mini_vllm/cuda/gpt2_pd_engine.hpp#L89)。以下为当前文件的原样摘录。

```cpp
        PDStepResult result;
        try_handoff(result);
        // 最多保留一个待交接请求；D 忙时停止接纳 P 请求，形成有界背压。
        if (!prefilling_ && !pending_ && !waiting_.empty()) {
            auto next = waiting_.front();
            if (p_blocks_.ensure_capacity(*next->sequence, next->sequence->num_prompt_tokens())) {
                waiting_.pop_front();
                prefilling_ = next;
                next->stage = PDStage::Prefilling;
                next->sequence->set_status(SequenceStatus::Running);
            }
        }
```

系统最多保留一个待交接请求。只要 pending 未被 D 接纳，P 就不继续从 waiting 接新任务，避免累积越来越多已完成 Prefill 的 KV。

这条限制给中间流水段提供了明确上界，但不等于整个服务具备完整流量控制。外部 waiting 队列仍可继续接收请求，`requests_` 也保留完成 handle，没有实现长期服务需要的结果清理接口。

D 仍继续推进已有 Decode，请求结束释放预留页后，下一轮 `try_handoff` 就可能成功。容量受限测试用这个路径验证不会因为接纳过多请求而卡死。

## 14. 迁移测试怎样防止“碰巧 Token 相同”

源码：[dev/cuda/test_gpt2_pd_engine.cu，第 33—55 行](../dev/cuda/test_gpt2_pd_engine.cu#L33)。以下为当前文件的原样摘录。

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
```

测试故意先在 D 上用 blocker 占一页，使源目标首物理页不同。它不只比较 greedy ID，还比较后续 Decode 的真实词表 logits，要求迁移后与继续在原卡生成一致。

三项断言分别证明不同内容：

- `source first page != target first page`：确实测试了页号重映射。
- `payload_bytes`：传输页数与模型布局相符。
- `error < 1e-5`：相同输入、相同精度下两端后续数值一致。

`copy` 不负责释放源页，测试专门检查源 free 数量未变。源是否还能使用由 Engine 的所有权交接决定，而不是 memcpy 函数暗中决定。

## 15. 整条流水线还要验证哪些场景

测试 Prompt 长度为 1、16、17、31、32、33，覆盖页边界；普通 D 容量与仅 3 页的受限容量都运行。还检查：

| 场景 | 期望 |
| --- | --- |
| max_new_tokens=1 | P 首 Token 完成后直接结束，不迁移 |
| 首 Token 为 EOS | 同样不迁移 |
| D 容量不足以接纳下一请求 | pending 保留，D 继续推进并释放页 |
| 重复请求 ID | 入口拒绝 |
| 单请求超过池容量 | 提前拒绝，避免永久等待 |
| 所有请求完成 | P/D 页均归还，输出与 CPU/单 GPU 参考一致 |

测试循环还有轮数上限，避免一次调度逻辑回归让测试无限挂起。但有轮数保护不代表实现了任意输入下的形式化活性证明。

## 16. 如何解释当前双卡反而更慢

请看 [任务 09—11 的证据汇总](../benchmark/results/task09_11/README.md) 和文末历史记录。当前短 GPT-2 负载下，Graph 模式单卡约 9.876 ms，PD 约 18.493 ms，交接累计约 7.018 ms。

可以用这个时间分解理解：

```text
PD 总时间 ≈ 不可重叠的启动/排空
          + 同轮 P/D 计算的较慢一侧
          + 同步 KV 迁移
          + CPU 调度、线程、元数据和同步开销
```

这只是解释框架，实际耗时要从测量得到，不能直接把所有阶段时长相加后当作严谨预测。不同请求与不同轮次存在重叠和等待。

当前模型小、输出短，单卡计算已经很快；host staging、页拷贝、同步和双端协调相对明显。两张卡都工作并不自动带来加速。

此外目前只比较单卡与 PD，没有提供双副本数据并行基线，因此不能据此证明 PD 是这两张卡上最优的服务方式。

## 17. 本阶段学到什么才算完成

你应该能够在不运行代码时，独立说出一次交接至少包含：源 KV 数值、目标页映射、computed、Prompt/生成 ID、采样参数、请求身份与阶段。仅复制 K/V 裸字节还不足以恢复生成状态。

你也应能指出当前边界：同进程双模型副本、同步 host staging、单个 P 活跃请求、有界 pending、D 预留上限、没有 Tensor Parallel/跨机通信/异步 KV overlap/服务故障恢复。

这些边界让简历描述准确，也给面试时的后续优化讨论提供依据。现阶段先掌握这条完整路径，不需要立即再增加通信库或更多调度策略。

## 18. 练习与参考答案

**题 1：Prompt=33，生成 6 个，P 传多少页，D 最多预留多少页？**

答案：P 已计算 Prompt，传 `ceil(33/16)=3` 页；D 最大处理 `33+6−1=38` 个，预留 `ceil(38/16)=3` 页。两者本例恰好相同，但 Prompt=32、生成 6 时会分别为 2 页和 3 页。

**题 2：P 已追加首 Token，为什么 D append 前先 mark_computed(Prompt 长度)？**

答案：目标初始只有 Prompt。先标记这些输入的 KV 已迁入，再追加尚未计算 KV 的首 Token，恢复 pending=1 的 Decode 状态。

**题 3：源页号 `[0,2]`，目标页号 `[5,1]`，能整体 memcpy 到目标起始页 0 吗？**

答案：不能。必须按逻辑页 0→物理 5、逻辑页 1→物理 1 分别写入；两个池的物理编号互不对应。

**题 4：传输中途出错，为什么不能直接先释放 P 页？**

答案：D 可能没有完整有效 KV，源是仍保留的请求状态。当前 catch 回收目标并抛错，但不承诺自动恢复服务。

**题 5：日志显示 9 轮 concurrent submissions，能写“9 轮完全重叠”吗？**

答案：不能，必须用实际 GPU 时间轴证明重叠时长。host 分支只证明尝试并发提交。

**题 6：P 处理完首 Token 为 EOS 的请求，也应迁移一次验证链路吗？**

答案：正常生成不需要，直接结束最合理。迁移链路由独立测试覆盖，不应给已完成请求增加无意义工作。

读完后回到 [中文学习手册](paged_inference_learning_zh.md)，按“请求生命周期 → 地址映射 → 数值执行 → 优化验证 → PD 交接”复述整个项目。

---

## 原开发记录与阶段实验

以下保留本任务开发时的目标、验收与测量记录。涉及后续任务改动的行为，以前面的当前源码精读为准；旧性能数据只代表记录中的配置。

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
