#ifndef MINI_VLLM_CUDA_GPT2_PD_ENGINE_HPP
#define MINI_VLLM_CUDA_GPT2_PD_ENGINE_HPP

#include "gpt2_cuda_model_runner.cuh"
#include "paged_attention.cuh"

#include <deque>
#include <future>
#include <unordered_map>

namespace mini_vllm { namespace cuda {

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

struct PDStepResult {
    std::size_t prefill_tokens = 0;
    std::size_t decode_tokens = 0;
    bool concurrent_submissions = false;
    std::vector<std::uint64_t> handed_off;
    std::vector<std::uint64_t> sampled_request_ids;
    std::vector<int> sampled_token_ids;
};

// 同进程、双 GPU、双模型副本：GPU P 只做 Prompt，GPU D 只做后续 Decode。
// 第一版使用同步 host staging 交接；计算阶段可同时提交 P(B) 与 D(A)。
class GPT2PDEngine {
public:
    GPT2PDEngine(GPT2CudaConfig config, const float* parameters,
                 std::size_t num_parameters, int prefill_device, int decode_device,
                 std::size_t prefill_blocks, std::size_t decode_blocks,
                 std::size_t prefill_token_budget, std::size_t max_decode_sequences,
                 std::size_t max_context_length)
        : p_blocks_(prefill_blocks, kPagedAttentionPageSize),
          d_blocks_(decode_blocks, kPagedAttentionPageSize),
          d_scheduler_({max_decode_sequences, max_decode_sequences}, d_blocks_),
          p_runner_(on_device(config, prefill_device, decode_device), parameters,
                    num_parameters, p_blocks_, 1, prefill_token_budget, max_context_length),
          d_runner_(on_device(config, decode_device, prefill_device), parameters,
                    num_parameters, d_blocks_, max_decode_sequences,
                    max_decode_sequences, max_context_length),
          p_budget_(prefill_token_budget), max_decode_(max_decode_sequences),
          max_context_(max_context_length), vocab_size_(config.vocab_size) {}

    std::shared_ptr<PDRequest> add_request(
        std::uint64_t id, std::vector<int> prompt, SamplingParams sampling) {
        if (requests_.count(id)) throw std::invalid_argument("duplicate PD request id");
        if (prompt.empty() || sampling.max_new_tokens == 0 ||
            prompt.size() > max_context_ ||
            sampling.max_new_tokens - 1 > max_context_ - prompt.size()) {
            throw std::invalid_argument("invalid PD request length");
        }
        for (int token : prompt) {
            if (token < 0 || token >= vocab_size_)
                throw std::out_of_range("PD prompt token outside vocabulary");
        }
        // 提前拒绝单请求也装不下的输入；D 预留生成上限，避免多请求扩页死锁。
        if (p_blocks_.blocks_needed(prompt.size()) > p_blocks_.num_blocks() ||
            (sampling.max_new_tokens > 1 &&
             d_blocks_.blocks_needed(prompt.size() + sampling.max_new_tokens - 1) >
                 d_blocks_.num_blocks())) {
            throw std::out_of_range("PD request exceeds a worker KV pool");
        }
        auto request = std::make_shared<PDRequest>();
        request->sequence = std::make_shared<Sequence>(id, std::move(prompt), sampling);
        request->sampling = sampling;
        requests_.emplace(id, request);
        waiting_.push_back(request);
        return request;
    }

    bool is_finished() const {
        return waiting_.empty() && !prefilling_ && !pending_ && d_scheduler_.is_finished();
    }
    std::size_t prefill_free_blocks() const { return p_blocks_.num_free_blocks(); }
    std::size_t decode_free_blocks() const { return d_blocks_.num_free_blocks(); }

    PDStepResult step() {
        if (is_finished()) throw std::logic_error("cannot step a finished PD engine");
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
        SchedulerOutput p_output;
        if (prefilling_) {
            const auto count = std::min(p_budget_, prefilling_->sequence->pending_tokens());
            p_output.items.push_back({prefilling_->sequence, ExecutionPhase::Prefill, count});
            p_output.num_batched_tokens = count;
        }
        const auto d_output = d_scheduler_.schedule();
        result.prefill_tokens = p_output.num_batched_tokens;
        result.decode_tokens = d_output.num_batched_tokens;
        if (p_output.items.empty() && d_output.items.empty())
            throw std::runtime_error("PD made no progress");

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
        if (!d_output.items.empty()) {
            d_scheduler_.commit(d_output, d_samples);
            for (std::size_t i = 0; i < d_output.items.size(); ++i) {
                auto sequence = d_output.items[i].sequence;
                record_sample(result, sequence->request_id(), d_samples[i]);
                if (sequence->is_finished())
                    requests_.at(sequence->request_id())->stage = PDStage::Finished;
            }
        }
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
        }
        p_blocks_.validate();
        d_blocks_.validate();
        return result;
    }

private:
    static GPT2CudaConfig on_device(GPT2CudaConfig config, int device, int other) {
        if (device < 0 || other < 0 || device == other)
            throw std::invalid_argument("PD requires two distinct GPU ids");
        config.device_id = device;
        return config;
    }
    static void record_sample(PDStepResult& result, std::uint64_t id, int sample) {
        result.sampled_request_ids.push_back(id);
        result.sampled_token_ids.push_back(sample);
    }

    void try_handoff(PDStepResult& result) {
        if (!pending_ || d_scheduler_.num_waiting() + d_scheduler_.num_running() >= max_decode_)
            return;
        auto& source = *pending_->sequence;
        std::vector<int> prompt(source.token_ids().begin(),
                                source.token_ids().begin() + source.num_prompt_tokens());
        auto target = std::make_shared<Sequence>(source.request_id(), std::move(prompt), pending_->sampling);
        const auto reserve_tokens = source.num_prompt_tokens() + pending_->sampling.max_new_tokens - 1;
        if (!d_blocks_.ensure_capacity(*target, reserve_tokens)) return;
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

    BlockManager p_blocks_, d_blocks_;
    Scheduler d_scheduler_;
    GPT2CudaModelRunner p_runner_, d_runner_;
    std::size_t p_budget_, max_decode_, max_context_;
    int vocab_size_;
    std::unordered_map<std::uint64_t, std::shared_ptr<PDRequest>> requests_;
    std::deque<std::shared_ptr<PDRequest>> waiting_;
    std::shared_ptr<PDRequest> prefilling_, pending_;
};

}} // namespace mini_vllm::cuda
#endif
