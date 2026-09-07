#include "gpt2_engine.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace mini_vllm;

static const char* phase_name(ExecutionPhase phase) {
    return phase == ExecutionPhase::Prefill ? "Prefill" : "Decode";
}

static std::vector<int> prompt(std::size_t length, int seed) {
    std::vector<int> tokens(length);
    tokens[0] = 50256;
    for (std::size_t i = 1; i < length; ++i) {
        tokens[i] =
            100 + (seed * 3571 + static_cast<int>(i) * 7919) % 50000;
    }
    return tokens;
}

int main() {
    GPT2 model{};
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    GPT2Engine engine(
        model,
        /*num_kv_blocks=*/8,
        {/*max_num_sequences=*/3, /*max_num_batched_tokens=*/8},
        /*max_context_length=*/64);

    std::vector<std::shared_ptr<Sequence>> requests;
    requests.push_back(engine.add_request(
        1, prompt(6, 1), SamplingParams{/*max_new_tokens=*/2, -1, false}));
    requests.push_back(engine.add_request(
        2, prompt(11, 2), SamplingParams{/*max_new_tokens=*/2, -1, false}));

    for (int step = 1; !engine.is_finished(); ++step) {
        if (step == 3) {
            requests.push_back(engine.add_request(
                3, prompt(3, 3),
                SamplingParams{/*max_new_tokens=*/2, -1, false}));
        }

        const EngineStepResult result = engine.step();
        std::cout << "轮次 " << step
                  << "：调度 Token=" << result.num_batched_tokens
                  << "，微批次数=" << result.num_micro_batches
                  << "，任务=[";
        for (std::size_t i = 0; i < result.request_ids.size(); ++i) {
            if (i != 0) std::cout << ", ";
            std::cout << "请求" << result.request_ids[i] << ':'
                      << phase_name(result.phases[i]) << 'x'
                      << result.scheduled_token_counts[i]
                      << "，采样=" << result.sampled_token_ids[i];
        }
        std::cout << "]，空闲 Block="
                  << result.free_blocks_after_commit << '\n';
    }

    for (const auto& request : requests) {
        std::cout << "请求 " << request->request_id() << " 输出 Token：";
        for (std::size_t i = request->num_prompt_tokens();
             i < request->num_tokens(); ++i) {
            std::cout << request->token_ids()[i] << ' ';
        }
        std::cout << '\n';
    }

    gpt2_free(&model);
    return 0;
}
