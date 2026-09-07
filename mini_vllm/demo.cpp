#include "scheduler.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mini_vllm;

static const char* phase_name(ExecutionPhase phase) {
    return phase == ExecutionPhase::Prefill ? "prefill" : "decode";
}

static std::shared_ptr<Sequence> make_request(std::uint64_t id,
                                               std::size_t prompt_length,
                                               std::size_t max_new_tokens) {
    std::vector<int> tokens(prompt_length, static_cast<int>(id));
    return std::make_shared<Sequence>(
        id, std::move(tokens), SamplingParams{max_new_tokens, -1, false});
}

int main() {
    BlockManager block_manager(/*num_blocks=*/12, /*block_size=*/4);
    Scheduler scheduler({/*max_num_sequences=*/3,
                         /*max_num_batched_tokens=*/8},
                        block_manager);

    scheduler.add(make_request(1, 6, 3));
    scheduler.add(make_request(2, 11, 2));

    for (int step = 1; !scheduler.is_finished(); ++step) {
        // Demonstrate that a new request may enter while older requests decode.
        if (step == 3) scheduler.add(make_request(3, 3, 2));

        const SchedulerOutput output = scheduler.schedule();
        if (output.items.empty()) {
            std::cerr << "scheduler made no progress (KV blocks may be exhausted)\n";
            return 1;
        }

        std::cout << "step=" << step
                  << " batched_tokens=" << output.num_batched_tokens
                  << " free_blocks_before=" << block_manager.num_free_blocks()
                  << " work=[";
        std::vector<int> sampled_tokens;
        for (std::size_t i = 0; i < output.items.size(); ++i) {
            const ScheduledItem& item = output.items[i];
            const Sequence& sequence = *item.sequence;
            if (i != 0) std::cout << ", ";
            std::cout << "req" << sequence.request_id() << ':'
                      << phase_name(item.phase) << 'x' << item.num_scheduled_tokens;

            const bool completes_input =
                item.num_scheduled_tokens == sequence.pending_tokens();
            sampled_tokens.push_back(
                completes_input
                    ? 1000 + static_cast<int>(sequence.request_id() * 10) +
                          static_cast<int>(sequence.num_completion_tokens())
                    : -1);
        }
        std::cout << "]\n";

        scheduler.commit(output, sampled_tokens);
        std::cout << "       waiting=" << scheduler.num_waiting()
                  << " running=" << scheduler.num_running()
                  << " free_blocks_after=" << block_manager.num_free_blocks()
                  << "\n";
    }
}
