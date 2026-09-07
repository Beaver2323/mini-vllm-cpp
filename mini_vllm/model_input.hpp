#ifndef MINI_VLLM_MODEL_INPUT_HPP
#define MINI_VLLM_MODEL_INPUT_HPP

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
    // Packed Prefill 中第 i 个请求的 Token 范围是
    // [query_start_locations[i], query_start_locations[i + 1])。
    std::vector<std::size_t> query_start_locations;
    std::size_t max_blocks_per_sequence = 0;

    std::size_t batch_size() const { return token_ids.size(); }
};

inline int model_input_checked_int(std::size_t value, const char* message) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::overflow_error(message);
    }
    return static_cast<int>(value);
}

inline ModelInput prepare_packed_model_input(
    const SchedulerOutput& output, const BlockManager& block_manager,
    std::size_t max_context_length,
    std::size_t max_blocks_per_sequence, std::size_t num_kv_pages) {
    ModelInput input;
    input.max_blocks_per_sequence = max_blocks_per_sequence;
    input.query_start_locations.reserve(output.items.size() + 1);

    for (std::size_t item_index = 0;
         item_index < output.items.size(); ++item_index) {
        const ScheduledItem& item = output.items[item_index];
        if (item.sequence == nullptr || item.num_scheduled_tokens == 0) {
            throw std::invalid_argument("scheduled item is invalid");
        }
        const Sequence& sequence = *item.sequence;
        if (sequence.block_table().size() > max_blocks_per_sequence) {
            throw std::out_of_range(
                "sequence block table exceeds ModelRunner capacity");
        }
        input.query_start_locations.push_back(input.batch_size());

        for (std::size_t offset = 0;
             offset < item.num_scheduled_tokens; ++offset) {
            const std::size_t position =
                sequence.num_computed_tokens() + offset;
            if (position >= sequence.num_tokens() ||
                position >= max_context_length) {
                throw std::out_of_range(
                    "scheduled token exceeds sequence or context capacity");
            }
            const int physical_block =
                block_manager.block_id_for_token(sequence, position);
            if (physical_block < 0 ||
                static_cast<std::size_t>(physical_block) >= num_kv_pages) {
                throw std::out_of_range(
                    "sequence references an invalid physical KV block");
            }

            input.token_ids.push_back(sequence.token_ids()[position]);
            input.positions.push_back(model_input_checked_int(
                position, "token position is too large"));
            input.context_lengths.push_back(model_input_checked_int(
                position + 1, "context length is too large"));
            const std::size_t physical_slot =
                static_cast<std::size_t>(physical_block) *
                    block_manager.block_size() +
                block_manager.slot_for_token(position);
            input.slot_mapping.push_back(model_input_checked_int(
                physical_slot, "physical KV slot is too large"));
            input.request_ids.push_back(sequence.request_id());
            input.scheduled_item_indices.push_back(item_index);

            const std::size_t row_start = input.block_tables.size();
            input.block_tables.resize(
                row_start + max_blocks_per_sequence, -1);
            std::copy(
                sequence.block_table().begin(),
                sequence.block_table().end(),
                input.block_tables.begin() +
                    static_cast<std::ptrdiff_t>(row_start));
        }
    }
    input.query_start_locations.push_back(input.batch_size());
    if (input.batch_size() == 0 ||
        input.batch_size() != output.num_batched_tokens) {
        throw std::logic_error(
            "packed ModelInput does not match scheduled token count");
    }
    return input;
}

inline ModelInput prepare_model_input(
    const SchedulerOutput& output, std::size_t micro_step,
    const BlockManager& block_manager, std::size_t max_context_length,
    std::size_t max_blocks_per_sequence, std::size_t num_kv_pages) {
    ModelInput input;
    input.max_blocks_per_sequence = max_blocks_per_sequence;

    for (std::size_t item_index = 0;
         item_index < output.items.size(); ++item_index) {
        const ScheduledItem& item = output.items[item_index];
        if (micro_step >= item.num_scheduled_tokens) continue;

        const Sequence& sequence = *item.sequence;
        const std::size_t position =
            sequence.num_computed_tokens() + micro_step;
        if (position >= sequence.num_tokens() ||
            position >= max_context_length) {
            throw std::out_of_range(
                "scheduled token exceeds sequence or context capacity");
        }

        const int physical_block =
            block_manager.block_id_for_token(sequence, position);
        if (physical_block < 0 ||
            static_cast<std::size_t>(physical_block) >= num_kv_pages) {
            throw std::out_of_range(
                "sequence references an invalid physical KV block");
        }
        if (sequence.block_table().size() > max_blocks_per_sequence) {
            throw std::out_of_range(
                "sequence block table exceeds ModelRunner capacity");
        }

        input.token_ids.push_back(sequence.token_ids()[position]);
        input.positions.push_back(model_input_checked_int(
            position, "token position is too large"));
        input.context_lengths.push_back(model_input_checked_int(
            position + 1, "context length is too large"));
        const std::size_t physical_slot =
            static_cast<std::size_t>(physical_block) *
                block_manager.block_size() +
            block_manager.slot_for_token(position);
        input.slot_mapping.push_back(model_input_checked_int(
            physical_slot, "physical KV slot is too large"));
        input.request_ids.push_back(sequence.request_id());
        input.scheduled_item_indices.push_back(item_index);

        const std::size_t row_start = input.block_tables.size();
        input.block_tables.resize(
            row_start + max_blocks_per_sequence, -1);
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

} // namespace mini_vllm

#endif
