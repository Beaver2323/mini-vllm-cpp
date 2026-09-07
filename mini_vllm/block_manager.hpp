#ifndef MINI_VLLM_BLOCK_MANAGER_HPP
#define MINI_VLLM_BLOCK_MANAGER_HPP

#include "sequence.hpp"

#include <cstddef>
#include <deque>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mini_vllm {

struct Block {
    int id = -1;
    std::size_t ref_count = 0;
};

class BlockManager {
public:
    BlockManager(std::size_t num_blocks, std::size_t block_size)
        : block_size_(block_size), blocks_(num_blocks) {
        if (num_blocks == 0 || block_size == 0) {
            throw std::invalid_argument("num_blocks and block_size must be positive");
        }
        for (std::size_t i = 0; i < num_blocks; ++i) {
            blocks_[i].id = static_cast<int>(i);
            free_block_ids_.push_back(static_cast<int>(i));
        }
    }

    std::size_t block_size() const { return block_size_; }
    std::size_t num_blocks() const { return blocks_.size(); }
    std::size_t num_free_blocks() const { return free_block_ids_.size(); }
    std::size_t num_used_blocks() const { return blocks_.size() - free_block_ids_.size(); }

    std::size_t blocks_needed(std::size_t token_count) const {
        return token_count == 0 ? 0 : (token_count + block_size_ - 1) / block_size_;
    }

    bool can_ensure_capacity(const Sequence& sequence, std::size_t token_count) const {
        const std::size_t required = blocks_needed(token_count);
        if (required <= sequence.block_table().size()) {
            return true;
        }
        return required - sequence.block_table().size() <= free_block_ids_.size();
    }

    bool ensure_capacity(Sequence& sequence, std::size_t token_count) {
        const std::size_t required = blocks_needed(token_count);
        if (required <= sequence.block_table().size()) {
            return true;
        }
        const std::size_t additional = required - sequence.block_table().size();
        if (additional > free_block_ids_.size()) {
            return false;
        }
        if (additional > 0 && sequence.block_table().empty()) {
            if (!live_sequences_.insert(sequence.request_id()).second) {
                throw std::logic_error("sequence allocation state is inconsistent");
            }
        }
        for (std::size_t i = 0; i < additional; ++i) {
            const int block_id = free_block_ids_.front();
            free_block_ids_.pop_front();
            Block& block = blocks_.at(static_cast<std::size_t>(block_id));
            if (block.ref_count != 0) {
                throw std::logic_error("free list contains a referenced block");
            }
            block.ref_count = 1;
            sequence.block_table().push_back(block_id);
        }
        return true;
    }

    void release(Sequence& sequence) {
        if (sequence.block_table().empty() ||
            live_sequences_.erase(sequence.request_id()) != 1) {
            throw std::logic_error("sequence blocks are absent or were already released");
        }
        // Validate the entire table before changing any reference count so release
        // either succeeds completely or fails without partially returning blocks.
        for (int block_id : sequence.block_table()) {
            const std::size_t index = static_cast<std::size_t>(block_id);
            if (block_id < 0 || index >= blocks_.size() ||
                blocks_[index].ref_count == 0) {
                live_sequences_.insert(sequence.request_id());
                throw std::logic_error("sequence block table is corrupt");
            }
        }
        for (auto it = sequence.block_table().rbegin();
             it != sequence.block_table().rend(); ++it) {
            Block& block = blocks_.at(static_cast<std::size_t>(*it));
            if (block.ref_count == 0) {
                throw std::logic_error("block reference count underflow");
            }
            --block.ref_count;
            if (block.ref_count == 0) {
                free_block_ids_.push_back(block.id);
            }
        }
        sequence.block_table().clear();
    }

    int block_id_for_token(const Sequence& sequence, std::size_t token_index) const {
        const std::size_t logical_block = token_index / block_size_;
        if (logical_block >= sequence.block_table().size()) {
            throw std::out_of_range("token has no allocated KV cache block");
        }
        return sequence.block_table()[logical_block];
    }

    std::size_t slot_for_token(std::size_t token_index) const {
        return token_index % block_size_;
    }

    void validate() const {
        std::vector<bool> on_free_list(blocks_.size(), false);
        for (int block_id : free_block_ids_) {
            const std::size_t index = static_cast<std::size_t>(block_id);
            if (index >= blocks_.size() || on_free_list[index]) {
                throw std::logic_error("invalid or duplicate block on free list");
            }
            on_free_list[index] = true;
            if (blocks_[index].ref_count != 0) {
                throw std::logic_error("referenced block appears on free list");
            }
        }
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            if (!on_free_list[i] && blocks_[i].ref_count == 0) {
                throw std::logic_error("unreferenced block is missing from free list");
            }
        }
    }

private:
    std::size_t block_size_;
    std::vector<Block> blocks_;
    std::deque<int> free_block_ids_;
    std::unordered_set<std::uint64_t> live_sequences_;
};

} // namespace mini_vllm

#endif
