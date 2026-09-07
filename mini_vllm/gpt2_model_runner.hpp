#ifndef MINI_VLLM_GPT2_MODEL_RUNNER_HPP
#define MINI_VLLM_GPT2_MODEL_RUNNER_HPP

#ifndef GPT2_PAGED_INFERENCE_NO_MAIN
#define GPT2_PAGED_INFERENCE_NO_MAIN
#endif
#include "../train_gpt2.cpp"
#include "scheduler.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mini_vllm {

// 一个 ModelInput 表示 ModelRunner 内部的一个单 Token 微批次。
// Chunked Prefill 会被拆成多个微批次；每个微批次仍可同时包含多个请求。
struct ModelInput {
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> context_lengths;
    std::vector<int> slot_mapping;
    std::vector<int> block_tables;
    std::vector<std::uint64_t> request_ids;
    std::vector<std::size_t> scheduled_item_indices;
    std::size_t max_blocks_per_sequence = 0;

    std::size_t batch_size() const { return token_ids.size(); }
};

class GPT2ModelRunner {
public:
    GPT2ModelRunner(
        GPT2& model, BlockManager& block_manager,
        std::size_t max_num_sequences, std::size_t max_context_length)
        : model_(model),
          block_manager_(block_manager),
          max_context_length_(max_context_length),
          max_blocks_per_sequence_(
              (max_context_length + PAGE_SIZE - 1) / PAGE_SIZE),
          kv_cache_pool_(
              checked_int(block_manager.num_blocks(), "too many KV blocks"),
              model.config.num_layers, model.config.num_heads,
              model.config.channels / model.config.num_heads),
          workspace_(
              model.config,
              checked_int(max_num_sequences, "too many active sequences"),
              checked_int(max_context_length, "context length is too large")) {
        if (model.params_memory == nullptr) {
            throw std::invalid_argument("GPT-2 model weights are not initialized");
        }
        if (block_manager.block_size() != PAGE_SIZE) {
            throw std::invalid_argument(
                "BlockManager block size must equal the KV cache page size");
        }
        if (max_num_sequences == 0 || max_context_length == 0 ||
            max_context_length >
                static_cast<std::size_t>(model.config.max_seq_len)) {
            throw std::invalid_argument("invalid ModelRunner capacity");
        }
    }

    GPT2ModelRunner(const GPT2ModelRunner&) = delete;
    GPT2ModelRunner& operator=(const GPT2ModelRunner&) = delete;

    std::vector<int> run(const SchedulerOutput& output) {
        if (output.items.empty()) {
            throw std::invalid_argument("ModelRunner received an empty schedule");
        }
        if (output.items.size() >
            static_cast<std::size_t>(workspace_.max_batch_size())) {
            throw std::out_of_range(
                "scheduled request count exceeds ModelRunner capacity");
        }

        std::size_t max_micro_steps = 0;
        for (const ScheduledItem& item : output.items) {
            if (item.sequence == nullptr || item.num_scheduled_tokens == 0) {
                throw std::invalid_argument("scheduled item is invalid");
            }
            max_micro_steps =
                std::max(max_micro_steps, item.num_scheduled_tokens);
        }

        std::vector<int> sampled_token_ids(output.items.size(), -1);
        last_model_inputs_.clear();
        last_model_inputs_.reserve(max_micro_steps);

        for (std::size_t micro_step = 0; micro_step < max_micro_steps;
             ++micro_step) {
            ModelInput input = prepare_model_input(output, micro_step);
            PageTable page_table(
                checked_int(input.batch_size(), "micro batch is too large"),
                checked_int(max_blocks_per_sequence_,
                            "page table width is too large"));
            page_table.context_lengths = input.context_lengths;
            page_table.block_tables = input.block_tables;

            gpt2_forward_inference_with_workspace(
                &model_, input.token_ids.data(), &kv_cache_pool_, &page_table,
                checked_int(input.batch_size(), "micro batch is too large"),
                &workspace_);

            const ActivationTensors& acts = workspace_.acts();
            const int padded_vocab_size = model_.config.padded_vocab_size;
            for (std::size_t row = 0; row < input.batch_size(); ++row) {
                const std::size_t item_index =
                    input.scheduled_item_indices[row];
                const Sequence& sequence =
                    *output.items[item_index].sequence;
                const std::size_t position =
                    static_cast<std::size_t>(input.positions[row]);
                if (position + 1 == sequence.num_tokens()) {
                    sampled_token_ids[item_index] = greedy_argmax(
                        acts.logits + row * padded_vocab_size);
                }
            }
            last_model_inputs_.push_back(std::move(input));
        }
        return sampled_token_ids;
    }

    const std::vector<ModelInput>& last_model_inputs() const {
        return last_model_inputs_;
    }

    const KVCachePool& kv_cache_pool() const { return kv_cache_pool_; }
    std::size_t workspace_num_activations() const {
        return workspace_.num_activations();
    }

private:
    static int checked_int(std::size_t value, const char* message) {
        if (value > static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
            throw std::overflow_error(message);
        }
        return static_cast<int>(value);
    }

    int greedy_argmax(const float* logits) const {
        if (logits == nullptr) {
            throw std::invalid_argument("logits pointer is null");
        }
        return static_cast<int>(
            std::max_element(
                logits, logits + model_.config.vocab_size) -
            logits);
    }

    ModelInput prepare_model_input(
        const SchedulerOutput& output, std::size_t micro_step) const {
        ModelInput input;
        input.max_blocks_per_sequence = max_blocks_per_sequence_;

        for (std::size_t item_index = 0;
             item_index < output.items.size(); ++item_index) {
            const ScheduledItem& item = output.items[item_index];
            if (micro_step >= item.num_scheduled_tokens) {
                continue;
            }
            const Sequence& sequence = *item.sequence;
            const std::size_t position =
                sequence.num_computed_tokens() + micro_step;
            if (position >= sequence.num_tokens() ||
                position >= max_context_length_) {
                throw std::out_of_range(
                    "scheduled token exceeds sequence or context capacity");
            }

            const int physical_block =
                block_manager_.block_id_for_token(sequence, position);
            if (physical_block < 0 ||
                physical_block >= kv_cache_pool_.num_pages) {
                throw std::out_of_range(
                    "sequence references an invalid physical KV block");
            }
            if (sequence.block_table().size() >
                max_blocks_per_sequence_) {
                throw std::out_of_range(
                    "sequence block table exceeds ModelRunner capacity");
            }

            input.token_ids.push_back(sequence.token_ids()[position]);
            input.positions.push_back(checked_int(
                position, "token position is too large"));
            input.context_lengths.push_back(checked_int(
                position + 1, "context length is too large"));
            const std::size_t physical_slot =
                static_cast<std::size_t>(physical_block) * PAGE_SIZE +
                block_manager_.slot_for_token(position);
            input.slot_mapping.push_back(checked_int(
                physical_slot, "physical KV slot is too large"));
            input.request_ids.push_back(sequence.request_id());
            input.scheduled_item_indices.push_back(item_index);

            const std::size_t row_start = input.block_tables.size();
            input.block_tables.resize(
                row_start + max_blocks_per_sequence_, -1);
            std::copy(
                sequence.block_table().begin(),
                sequence.block_table().end(),
                input.block_tables.begin() +
                    static_cast<std::ptrdiff_t>(row_start));
        }

        if (input.batch_size() == 0) {
            throw std::logic_error("ModelRunner produced an empty micro batch");
        }
        return input;
    }

    GPT2& model_;
    BlockManager& block_manager_;
    std::size_t max_context_length_;
    std::size_t max_blocks_per_sequence_;
    KVCachePool kv_cache_pool_;
    GPT2InferenceWorkspace workspace_;
    std::vector<ModelInput> last_model_inputs_;
};

} // namespace mini_vllm

#endif
