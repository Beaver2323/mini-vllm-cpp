#ifndef MINI_VLLM_SCHEDULER_HPP
#define MINI_VLLM_SCHEDULER_HPP

#include "block_manager.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mini_vllm {

struct SchedulerConfig {
    std::size_t max_num_sequences = 1;
    std::size_t max_num_batched_tokens = 1;
};

struct ScheduledItem {
    std::shared_ptr<Sequence> sequence;
    ExecutionPhase phase = ExecutionPhase::Prefill;
    std::size_t num_scheduled_tokens = 0;
};

struct SchedulerOutput {
    std::vector<ScheduledItem> items;
    std::size_t num_batched_tokens = 0;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config, BlockManager& block_manager)
        : config_(config), block_manager_(block_manager) {
        if (config_.max_num_sequences == 0 ||
            config_.max_num_batched_tokens == 0) {
            throw std::invalid_argument("scheduler limits must be positive");
        }
    }

    void add(const std::shared_ptr<Sequence>& sequence) {
        if (!sequence || sequence->status() != SequenceStatus::Waiting) {
            throw std::invalid_argument("scheduler accepts only non-null waiting sequences");
        }
        if (!known_request_ids_.insert(sequence->request_id()).second) {
            throw std::logic_error("duplicate request id");
        }
        waiting_.push_back(sequence);
    }

    bool is_finished() const { return waiting_.empty() && running_.empty(); }
    std::size_t num_waiting() const { return waiting_.size(); }
    std::size_t num_running() const { return running_.size(); }

    SchedulerOutput schedule() {
        SchedulerOutput output;

        // Advance active requests first. This protects decode inter-token latency and
        // also makes progress on a chunked prefill admitted in an earlier iteration.
        for (const auto& sequence : running_) {
            if (output.items.size() >= config_.max_num_sequences ||
                output.num_batched_tokens >= config_.max_num_batched_tokens) {
                break;
            }
            try_schedule(sequence, output);
        }

        // Admit new requests with the token budget left by active requests.
        while (!waiting_.empty() &&
               running_.size() < config_.max_num_sequences &&
               output.items.size() < config_.max_num_sequences &&
               output.num_batched_tokens < config_.max_num_batched_tokens) {
            const auto sequence = waiting_.front();
            if (!try_schedule(sequence, output)) {
                break; // FCFS admission: do not bypass a blocked head request.
            }
            waiting_.pop_front();
            sequence->set_status(SequenceStatus::Running);
            running_.push_back(sequence);
        }
        return output;
    }

    // sampled_token_ids is parallel to output.items. Use -1 only for an item whose
    // chunk does not finish all currently available input tokens.
    void commit(const SchedulerOutput& output,
                const std::vector<int>& sampled_token_ids) {
        if (sampled_token_ids.size() != output.items.size()) {
            throw std::invalid_argument("one sample marker is required per scheduled item");
        }
        std::vector<std::shared_ptr<Sequence>> finished;
        for (std::size_t i = 0; i < output.items.size(); ++i) {
            const ScheduledItem& item = output.items[i];
            Sequence& sequence = *item.sequence;
            sequence.mark_computed(item.num_scheduled_tokens);
            if (sequence.pending_tokens() != 0) {
                if (sampled_token_ids[i] != -1) {
                    throw std::logic_error("partial prefill must not produce a sampled token");
                }
                continue;
            }
            const int sampled_token = sampled_token_ids[i];
            if (sampled_token < 0) {
                throw std::logic_error("completed model input requires a sampled token");
            }
            sequence.append_token(sampled_token);
            if (sequence.should_finish_after(sampled_token)) {
                sequence.set_status(SequenceStatus::Finished);
                block_manager_.release(sequence);
                finished.push_back(item.sequence);
            }
        }
        for (const auto& sequence : finished) {
            running_.erase(std::remove(running_.begin(), running_.end(), sequence),
                           running_.end());
        }
        block_manager_.validate();
    }

private:
    bool try_schedule(const std::shared_ptr<Sequence>& sequence,
                      SchedulerOutput& output) {
        if (sequence->pending_tokens() == 0) {
            throw std::logic_error("sequence has no input token awaiting model execution");
        }
        const std::size_t budget =
            config_.max_num_batched_tokens - output.num_batched_tokens;
        const std::size_t count = std::min(sequence->pending_tokens(), budget);
        const std::size_t target = sequence->num_computed_tokens() + count;
        if (!block_manager_.ensure_capacity(*sequence, target)) {
            return false;
        }
        output.items.push_back(
            {sequence, sequence->is_prefill() ? ExecutionPhase::Prefill
                                              : ExecutionPhase::Decode,
             count});
        output.num_batched_tokens += count;
        return true;
    }

    SchedulerConfig config_;
    BlockManager& block_manager_;
    std::deque<std::shared_ptr<Sequence>> waiting_;
    std::vector<std::shared_ptr<Sequence>> running_;
    std::unordered_set<std::uint64_t> known_request_ids_;
};

} // namespace mini_vllm

#endif
