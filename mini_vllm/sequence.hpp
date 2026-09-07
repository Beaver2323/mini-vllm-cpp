#ifndef MINI_VLLM_SEQUENCE_HPP
#define MINI_VLLM_SEQUENCE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_vllm {

enum class SequenceStatus { Waiting, Running, Finished };
enum class ExecutionPhase { Prefill, Decode };

struct SamplingParams {
    std::size_t max_new_tokens = 1;
    int eos_token_id = -1;
    bool ignore_eos = false;
};

class Sequence {
public:
    Sequence(std::uint64_t request_id, std::vector<int> prompt_tokens,
             SamplingParams sampling_params)
        : request_id_(request_id), token_ids_(std::move(prompt_tokens)),
          num_prompt_tokens_(token_ids_.size()), sampling_params_(sampling_params) {
        if (token_ids_.empty()) {
            throw std::invalid_argument("a request must contain at least one prompt token");
        }
        if (sampling_params_.max_new_tokens == 0) {
            throw std::invalid_argument("max_new_tokens must be positive");
        }
    }

    std::uint64_t request_id() const { return request_id_; }
    SequenceStatus status() const { return status_; }
    void set_status(SequenceStatus status) { status_ = status; }

    const std::vector<int>& token_ids() const { return token_ids_; }
    std::size_t num_tokens() const { return token_ids_.size(); }
    std::size_t num_prompt_tokens() const { return num_prompt_tokens_; }
    std::size_t num_completion_tokens() const {
        return token_ids_.size() - num_prompt_tokens_;
    }
    std::size_t num_computed_tokens() const { return num_computed_tokens_; }
    std::size_t pending_tokens() const {
        return token_ids_.size() - num_computed_tokens_;
    }
    bool is_prefill() const { return num_computed_tokens_ < num_prompt_tokens_; }
    bool is_finished() const { return status_ == SequenceStatus::Finished; }

    void mark_computed(std::size_t count) {
        if (count > pending_tokens()) {
            throw std::logic_error("cannot compute tokens that are not present in the sequence");
        }
        num_computed_tokens_ += count;
    }

    void append_token(int token_id) {
        if (num_computed_tokens_ != token_ids_.size()) {
            throw std::logic_error("a sampled token can only follow fully computed input tokens");
        }
        token_ids_.push_back(token_id);
    }

    bool should_finish_after(int sampled_token) const {
        const bool hit_eos = !sampling_params_.ignore_eos &&
                             sampling_params_.eos_token_id >= 0 &&
                             sampled_token == sampling_params_.eos_token_id;
        return hit_eos || num_completion_tokens() >= sampling_params_.max_new_tokens;
    }

    std::vector<int>& block_table() { return block_table_; }
    const std::vector<int>& block_table() const { return block_table_; }

private:
    std::uint64_t request_id_;
    std::vector<int> token_ids_;
    std::size_t num_prompt_tokens_ = 0;
    std::size_t num_computed_tokens_ = 0;
    SamplingParams sampling_params_;
    SequenceStatus status_ = SequenceStatus::Waiting;
    std::vector<int> block_table_;
};

} // namespace mini_vllm

#endif
