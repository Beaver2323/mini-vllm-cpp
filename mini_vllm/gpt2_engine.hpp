#ifndef MINI_VLLM_GPT2_ENGINE_HPP
#define MINI_VLLM_GPT2_ENGINE_HPP

#include "gpt2_model_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_vllm {

struct EngineStepResult {
    std::size_t num_batched_tokens = 0;
    std::size_t num_micro_batches = 0;
    std::size_t free_blocks_before_schedule = 0;
    std::size_t free_blocks_after_schedule = 0;
    std::size_t free_blocks_after_commit = 0;
    std::vector<std::uint64_t> request_ids;
    std::vector<ExecutionPhase> phases;
    std::vector<std::size_t> scheduled_token_counts;
    std::vector<int> sampled_token_ids;
    std::vector<std::vector<int>> block_tables_before_commit;
};

class GPT2Engine {
public:
    GPT2Engine(
        GPT2& model, std::size_t num_kv_blocks,
        SchedulerConfig scheduler_config, std::size_t max_context_length,
        bool enable_prefix_cache = false)
        : block_manager_(
              num_kv_blocks, PAGE_SIZE, enable_prefix_cache),
          scheduler_(scheduler_config, block_manager_),
          model_runner_(
              model, block_manager_, scheduler_config.max_num_sequences,
              max_context_length),
          vocab_size_(model.config.vocab_size),
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

    EngineStepResult step() {
        if (scheduler_.is_finished()) {
            throw std::logic_error("cannot step a finished engine");
        }

        EngineStepResult result;
        result.free_blocks_before_schedule =
            block_manager_.num_free_blocks();
        SchedulerOutput output = scheduler_.schedule();
        result.free_blocks_after_schedule =
            block_manager_.num_free_blocks();
        if (output.items.empty()) {
            throw std::runtime_error(
                "scheduler made no progress; KV cache may be exhausted");
        }

        result.num_batched_tokens = output.num_batched_tokens;
        for (const ScheduledItem& item : output.items) {
            result.request_ids.push_back(item.sequence->request_id());
            result.phases.push_back(item.phase);
            result.scheduled_token_counts.push_back(
                item.num_scheduled_tokens);
            result.block_tables_before_commit.push_back(
                item.sequence->block_table());
        }

        result.sampled_token_ids = model_runner_.run(output);
        result.num_micro_batches =
            model_runner_.last_model_inputs().size();
        scheduler_.commit(output, result.sampled_token_ids);
        result.free_blocks_after_commit =
            block_manager_.num_free_blocks();
        return result;
    }

    bool is_finished() const { return scheduler_.is_finished(); }
    std::size_t num_waiting() const { return scheduler_.num_waiting(); }
    std::size_t num_running() const { return scheduler_.num_running(); }
    std::size_t num_free_blocks() const {
        return block_manager_.num_free_blocks();
    }
    std::size_t num_blocks() const { return block_manager_.num_blocks(); }
    std::size_t num_cached_blocks() const {
        return block_manager_.num_cached_blocks();
    }
    std::size_t prefix_cache_hit_blocks() const {
        return block_manager_.prefix_cache_hit_blocks();
    }

    const GPT2ModelRunner& model_runner() const { return model_runner_; }
    const BlockManager& block_manager() const { return block_manager_; }

private:
    BlockManager block_manager_;
    Scheduler scheduler_;
    GPT2ModelRunner model_runner_;
    int vocab_size_;
    std::size_t max_context_length_;
};

} // namespace mini_vllm

#endif
