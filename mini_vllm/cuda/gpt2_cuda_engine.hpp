#ifndef MINI_VLLM_CUDA_GPT2_CUDA_ENGINE_HPP
#define MINI_VLLM_CUDA_GPT2_CUDA_ENGINE_HPP

#include "gpt2_cuda_model_runner.cuh"
#include "paged_attention.cuh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_vllm {
namespace cuda {

struct CudaEngineStepResult {
    std::size_t num_batched_tokens = 0;
    std::size_t num_micro_batches = 0;
    std::vector<std::uint64_t> request_ids;
    std::vector<ExecutionPhase> phases;
    std::vector<std::size_t> scheduled_token_counts;
    std::vector<int> sampled_token_ids;
};

class GPT2CudaEngine {
public:
    GPT2CudaEngine(
        GPT2CudaConfig config, const float* host_parameters,
        std::size_t num_parameters, std::size_t num_kv_blocks,
        SchedulerConfig scheduler_config,
        std::size_t max_context_length)
        : block_manager_(num_kv_blocks, kPagedAttentionPageSize),
          scheduler_(scheduler_config, block_manager_),
          model_runner_(
              config, host_parameters, num_parameters, block_manager_,
              scheduler_config.max_num_sequences, max_context_length),
          vocab_size_(config.vocab_size),
          max_context_length_(max_context_length) {}

    std::shared_ptr<Sequence> add_request(
        std::uint64_t request_id, std::vector<int> prompt_tokens,
        SamplingParams sampling_params) {
        if (prompt_tokens.empty()) {
            throw std::invalid_argument("request prompt must not be empty");
        }
        if (sampling_params.max_new_tokens == 0) {
            throw std::invalid_argument("max_new_tokens must be positive");
        }
        const std::size_t additional_processed_tokens =
            sampling_params.max_new_tokens - 1;
        if (prompt_tokens.size() > max_context_length_ ||
            additional_processed_tokens >
                max_context_length_ - prompt_tokens.size()) {
            throw std::out_of_range(
                "request can exceed the configured context capacity");
        }
        for (int token_id : prompt_tokens) {
            if (token_id < 0 || token_id >= vocab_size_) {
                throw std::out_of_range(
                    "prompt token is outside the GPT-2 vocabulary");
            }
        }
        auto sequence = std::make_shared<Sequence>(
            request_id, std::move(prompt_tokens), sampling_params);
        scheduler_.add(sequence);
        return sequence;
    }

    CudaEngineStepResult step() {
        if (scheduler_.is_finished()) {
            throw std::logic_error("cannot step a finished CUDA engine");
        }
        SchedulerOutput output = scheduler_.schedule();
        if (output.items.empty()) {
            throw std::runtime_error(
                "CUDA scheduler made no progress; KV cache may be exhausted");
        }

        CudaEngineStepResult result;
        result.num_batched_tokens = output.num_batched_tokens;
        for (const ScheduledItem& item : output.items) {
            result.request_ids.push_back(item.sequence->request_id());
            result.phases.push_back(item.phase);
            result.scheduled_token_counts.push_back(
                item.num_scheduled_tokens);
        }
        result.sampled_token_ids = model_runner_.run(output);
        result.num_micro_batches =
            model_runner_.last_model_inputs().size();
        scheduler_.commit(output, result.sampled_token_ids);
        return result;
    }

    bool is_finished() const { return scheduler_.is_finished(); }
    std::size_t num_free_blocks() const {
        return block_manager_.num_free_blocks();
    }
    std::size_t num_blocks() const { return block_manager_.num_blocks(); }
    const GPT2CudaModelRunner& model_runner() const {
        return model_runner_;
    }

private:
    BlockManager block_manager_;
    Scheduler scheduler_;
    GPT2CudaModelRunner model_runner_;
    int vocab_size_;
    std::size_t max_context_length_;
};

} // namespace cuda
} // namespace mini_vllm

#endif
