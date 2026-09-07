#ifndef MINI_VLLM_BLOCK_MANAGER_HPP
#define MINI_VLLM_BLOCK_MANAGER_HPP

#include "sequence.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
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
    BlockManager(
        std::size_t num_blocks, std::size_t block_size,
        bool enable_prefix_cache = false)
        : block_size_(block_size), blocks_(num_blocks),
          prefix_cache_enabled_(enable_prefix_cache) {
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
    bool prefix_cache_enabled() const { return prefix_cache_enabled_; }
    std::size_t num_cached_blocks() const { return prefix_cache_.size(); }
    std::size_t prefix_cache_hit_blocks() const {
        return prefix_cache_hit_blocks_;
    }

    std::size_t blocks_needed(std::size_t token_count) const {
        return token_count == 0 ? 0 : (token_count + block_size_ - 1) / block_size_;
    }

    bool can_ensure_capacity(const Sequence& sequence, std::size_t token_count) const {
        const std::size_t required = blocks_needed(token_count);
        if (required <= sequence.block_table().size()) {
            return true;
        }
        const std::size_t additional =
            required - sequence.block_table().size();
        std::size_t evictable = 0;
        for (const auto& item : prefix_cache_) {
            if (blocks_.at(static_cast<std::size_t>(item.second.block_id))
                    .ref_count == 1) {
                ++evictable;
            }
        }
        return additional <= free_block_ids_.size() + evictable;
    }

    bool ensure_capacity(Sequence& sequence, std::size_t token_count) {
        const std::size_t required = blocks_needed(token_count);
        if (required <= sequence.block_table().size()) {
            return true;
        }
        const std::size_t additional = required - sequence.block_table().size();
        while (additional > free_block_ids_.size() &&
               evict_one_cached_block()) {}
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

    // 仅复用完整 Prompt Block，并始终留下至少一个 Token 重新计算 logits。
    // Cache Key 包含从 Prompt 开头到当前 Block 末尾的完整 Token 前缀，因为同一
    // Token Block 在不同历史上下文下产生的 K/V 并不相同。
    std::size_t apply_prefix_cache(Sequence& sequence) {
        if (!prefix_cache_enabled_ || sequence.num_computed_tokens() != 0 ||
            !sequence.block_table().empty()) {
            return 0;
        }
        const std::size_t cacheable_blocks =
            (sequence.num_prompt_tokens() - 1) / block_size_;
        std::size_t hits = 0;
        for (std::size_t logical_block = 0;
             logical_block < cacheable_blocks; ++logical_block) {
            const std::vector<int> key = prefix_key(sequence, logical_block);
            auto cached = prefix_cache_.find(key);
            if (cached == prefix_cache_.end()) break;
            Block& block = blocks_.at(
                static_cast<std::size_t>(cached->second.block_id));
            if (block.ref_count == 0) {
                throw std::logic_error("prefix cache references a free block");
            }
            ++block.ref_count;
            cached->second.last_used = ++cache_clock_;
            sequence.block_table().push_back(block.id);
            ++hits;
        }
        if (hits == 0) return 0;
        if (!live_sequences_.insert(sequence.request_id()).second) {
            throw std::logic_error("cached sequence allocation is inconsistent");
        }
        sequence.mark_computed(hits * block_size_);
        prefix_cache_hit_blocks_ += hits;
        return hits;
    }

    void cache_computed_prefix_blocks(const Sequence& sequence) {
        if (!prefix_cache_enabled_) return;
        const std::size_t prompt_cacheable =
            (sequence.num_prompt_tokens() - 1) / block_size_;
        const std::size_t computed_blocks =
            sequence.num_computed_tokens() / block_size_;
        const std::size_t count = std::min(
            {prompt_cacheable, computed_blocks,
             sequence.block_table().size()});
        for (std::size_t logical_block = 0;
             logical_block < count; ++logical_block) {
            const std::vector<int> key = prefix_key(sequence, logical_block);
            auto cached = prefix_cache_.find(key);
            if (cached != prefix_cache_.end()) {
                cached->second.last_used = ++cache_clock_;
                continue;
            }
            const int block_id = sequence.block_table()[logical_block];
            Block& block = blocks_.at(static_cast<std::size_t>(block_id));
            if (block.ref_count == 0) {
                throw std::logic_error("cannot cache an unreferenced block");
            }
            ++block.ref_count; // Prefix Cache 自身持有一个引用。
            prefix_cache_.emplace(
                std::move(key), CacheEntry{block_id, ++cache_clock_});
        }
    }

    void clear_prefix_cache() {
        for (const auto& item : prefix_cache_) {
            Block& block = blocks_.at(
                static_cast<std::size_t>(item.second.block_id));
            if (block.ref_count == 0) {
                throw std::logic_error("prefix cache reference underflow");
            }
            --block.ref_count;
            if (block.ref_count == 0) free_block_ids_.push_back(block.id);
        }
        prefix_cache_.clear();
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
        std::unordered_set<int> cached_block_ids;
        for (const auto& item : prefix_cache_) {
            const int block_id = item.second.block_id;
            const std::size_t index = static_cast<std::size_t>(block_id);
            if (block_id < 0 || index >= blocks_.size() ||
                blocks_[index].ref_count == 0 ||
                !cached_block_ids.insert(block_id).second) {
                throw std::logic_error("prefix cache metadata is corrupt");
            }
        }
    }

private:
    struct CacheEntry {
        int block_id = -1;
        std::uint64_t last_used = 0;
    };

    std::vector<int> prefix_key(
        const Sequence& sequence, std::size_t logical_block) const {
        const std::size_t end = (logical_block + 1) * block_size_;
        if (end > sequence.num_prompt_tokens()) {
            throw std::out_of_range("prefix block exceeds prompt");
        }
        return std::vector<int>(
            sequence.token_ids().begin(),
            sequence.token_ids().begin() +
                static_cast<std::ptrdiff_t>(end));
    }

    bool evict_one_cached_block() {
        auto victim = prefix_cache_.end();
        for (auto it = prefix_cache_.begin(); it != prefix_cache_.end(); ++it) {
            const Block& block = blocks_.at(
                static_cast<std::size_t>(it->second.block_id));
            if (block.ref_count == 1 &&
                (victim == prefix_cache_.end() ||
                 it->second.last_used < victim->second.last_used)) {
                victim = it;
            }
        }
        if (victim == prefix_cache_.end()) return false;
        Block& block = blocks_.at(
            static_cast<std::size_t>(victim->second.block_id));
        --block.ref_count;
        free_block_ids_.push_back(block.id);
        prefix_cache_.erase(victim);
        return true;
    }

    std::size_t block_size_;
    std::vector<Block> blocks_;
    std::deque<int> free_block_ids_;
    std::unordered_set<std::uint64_t> live_sequences_;
    bool prefix_cache_enabled_ = false;
    std::map<std::vector<int>, CacheEntry> prefix_cache_;
    std::uint64_t cache_clock_ = 0;
    std::size_t prefix_cache_hit_blocks_ = 0;
};

} // namespace mini_vllm

#endif
