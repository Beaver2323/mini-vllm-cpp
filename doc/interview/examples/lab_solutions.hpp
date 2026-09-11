#ifndef ZYF_INTERVIEW_LAB_SOLUTIONS_HPP
#define ZYF_INTERVIEW_LAB_SOLUTIONS_HPP
#include "mini_vllm/scheduler.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace interview_lab {
inline bool sample_ready(const mini_vllm::Sequence& sequence, std::size_t scheduled) {
    if (scheduled == 0 || scheduled > sequence.pending_tokens())
        throw std::invalid_argument("scheduled must be in [1, pending]");
    return scheduled == sequence.pending_tokens();
}
inline std::size_t physical_slot(const std::vector<int>& table,
                                 std::size_t position, std::size_t page_size) {
    if (page_size == 0) throw std::invalid_argument("page_size must be positive");
    const auto logical = position / page_size;
    if (logical >= table.size() || table[logical] < 0)
        throw std::out_of_range("position has no valid physical page");
    return static_cast<std::size_t>(table[logical]) * page_size + position % page_size;
}
inline std::vector<std::size_t> sample_rows(const mini_vllm::SchedulerOutput& output,
                                           const std::vector<std::size_t>& starts) {
    if (output.items.empty() || starts.size() != output.items.size() + 1 ||
        starts.front() != 0 || starts.back() != output.num_batched_tokens)
        throw std::invalid_argument("invalid packed boundaries");
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < output.items.size(); ++i) {
        const auto& item = output.items[i];
        if (!item.sequence || starts[i + 1] <= starts[i] ||
            starts[i + 1] - starts[i] != item.num_scheduled_tokens)
            throw std::invalid_argument("packed span does not match scheduled count");
        if (sample_ready(*item.sequence, item.num_scheduled_tokens))
            rows.push_back(starts[i + 1] - 1);
    }
    return rows;
}
inline void restore_after_handoff(mini_vllm::Sequence& target,
                                   const mini_vllm::Sequence& source) {
    const auto prompt = source.num_prompt_tokens();
    if (source.request_id() != target.request_id() ||
        source.num_computed_tokens() != prompt || source.num_completion_tokens() != 1 ||
        target.num_computed_tokens() != 0 || target.num_tokens() != prompt ||
        !std::equal(target.token_ids().begin(), target.token_ids().end(),
                    source.token_ids().begin()))
        throw std::invalid_argument("handoff requires matching prompt and one uncomputed sample");
    target.mark_computed(prompt);
    target.append_token(source.token_ids().back());
}
} // namespace interview_lab
#endif
