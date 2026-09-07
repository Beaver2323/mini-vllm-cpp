#include "../mini_vllm/gpt2_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace mini_vllm;

static std::vector<int> make_prompt(std::size_t length, int seed) {
    std::vector<int> tokens(length);
    tokens[0] = 50256;
    for (std::size_t i = 1; i < length; ++i) {
        tokens[i] =
            100 + (seed * 3571 + static_cast<int>(i) * 7919) % 50000;
    }
    return tokens;
}

static int reference_argmax(const float* logits, int vocab_size) {
    return static_cast<int>(
        std::max_element(logits, logits + vocab_size) - logits);
}

static void assert_step(
    const EngineStepResult& step,
    const std::vector<std::uint64_t>& request_ids,
    const std::vector<ExecutionPhase>& phases,
    const std::vector<std::size_t>& token_counts) {
    assert(step.request_ids == request_ids);
    assert(step.phases == phases);
    assert(step.scheduled_token_counts == token_counts);
}

int main() {
    GPT2 model{};
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    assert(model.acts_memory == nullptr);

    GPT2Engine engine(
        model,
        /*num_kv_blocks=*/4,
        {/*max_num_sequences=*/2, /*max_num_batched_tokens=*/8},
        /*max_context_length=*/64);

    bool rejected_invalid_token = false;
    try {
        engine.add_request(
            90, {model.config.vocab_size},
            SamplingParams{/*max_new_tokens=*/1, -1, false});
    } catch (const std::out_of_range&) {
        rejected_invalid_token = true;
    }
    assert(rejected_invalid_token);

    bool rejected_context_overflow = false;
    try {
        engine.add_request(
            91, make_prompt(64, 9),
            SamplingParams{/*max_new_tokens=*/2, -1, false});
    } catch (const std::out_of_range&) {
        rejected_context_overflow = true;
    }
    assert(rejected_context_overflow);

    {
        GPT2Engine blocked_engine(
            model,
            /*num_kv_blocks=*/1,
            {/*max_num_sequences=*/1, /*max_num_batched_tokens=*/32},
            /*max_context_length=*/64);
        blocked_engine.add_request(
            92, make_prompt(17, 9),
            SamplingParams{/*max_new_tokens=*/1, -1, false});
        bool rejected_stalled_schedule = false;
        try {
            blocked_engine.step();
        } catch (const std::runtime_error&) {
            rejected_stalled_schedule = true;
        }
        assert(rejected_stalled_schedule);
        assert(blocked_engine.num_free_blocks() ==
               blocked_engine.num_blocks());
    }

    auto request1 = engine.add_request(
        1, make_prompt(17, 1), SamplingParams{/*max_new_tokens=*/1, -1, false});

    const EngineStepResult step1 = engine.step();
    assert_step(
        step1, {1}, {ExecutionPhase::Prefill}, {8});
    assert(step1.num_micro_batches == 8);
    assert(step1.free_blocks_after_schedule == 3);

    auto request2 = engine.add_request(
        2, make_prompt(5, 2), SamplingParams{/*max_new_tokens=*/2, -1, false});

    const EngineStepResult step2 = engine.step();
    assert_step(
        step2, {1}, {ExecutionPhase::Prefill}, {8});

    const EngineStepResult step3 = engine.step();
    assert_step(
        step3, {1, 2},
        {ExecutionPhase::Prefill, ExecutionPhase::Prefill},
        {1, 5});
    assert(request1->is_finished());
    assert(!request2->is_finished());
    assert(step3.block_tables_before_commit[0] ==
           std::vector<int>({0, 1}));
    assert(step3.block_tables_before_commit[1] ==
           std::vector<int>({2}));
    assert(step3.free_blocks_after_commit == 3);

    auto request3 = engine.add_request(
        3, make_prompt(17, 3), SamplingParams{/*max_new_tokens=*/1, -1, false});

    const EngineStepResult step4 = engine.step();
    assert_step(
        step4, {2, 3},
        {ExecutionPhase::Decode, ExecutionPhase::Prefill},
        {1, 7});
    assert(request2->is_finished());
    assert(step4.block_tables_before_commit[1] ==
           std::vector<int>({3}));
    assert(step4.num_micro_batches == 7);

    const auto& mixed_inputs = engine.model_runner().last_model_inputs();
    assert(mixed_inputs.front().request_ids ==
           std::vector<std::uint64_t>({2, 3}));
    assert(mixed_inputs.front().positions == std::vector<int>({5, 0}));
    assert(mixed_inputs.front().context_lengths ==
           std::vector<int>({6, 1}));
    assert(mixed_inputs[1].request_ids ==
           std::vector<std::uint64_t>({3}));

    const EngineStepResult step5 = engine.step();
    assert_step(
        step5, {3}, {ExecutionPhase::Prefill}, {8});

    const EngineStepResult step6 = engine.step();
    assert_step(
        step6, {3}, {ExecutionPhase::Prefill}, {2});
    assert(step6.block_tables_before_commit[0] ==
           std::vector<int>({3, 1}));
    assert(request3->is_finished());
    assert(engine.is_finished());
    assert(engine.num_free_blocks() == engine.num_blocks());

    // 第 17 个 Token 跨到第二个 Block，并复用了 request1 释放的 Block 1。
    const auto& boundary_inputs = engine.model_runner().last_model_inputs();
    assert(boundary_inputs.size() == 2);
    assert(boundary_inputs[0].positions == std::vector<int>({15}));
    assert(boundary_inputs[0].slot_mapping ==
           std::vector<int>({3 * PAGE_SIZE + 15}));
    assert(boundary_inputs[1].positions == std::vector<int>({16}));
    assert(boundary_inputs[1].slot_mapping ==
           std::vector<int>({1 * PAGE_SIZE}));

    // ModelRunner 使用独立 Workspace，因此在进入 reference 前向前，
    // GPT2 自带的训练激活仍未分配。
    assert(model.acts_memory == nullptr);
    const std::size_t inference_activations =
        engine.model_runner().workspace_num_activations();

    const std::vector<std::shared_ptr<Sequence>> requests = {
        request1, request2, request3};
    std::size_t reference_length = 0;
    for (const auto& sequence : requests) {
        reference_length =
            std::max(reference_length, sequence->num_tokens());
    }
    std::vector<int> reference_tokens(
        requests.size() * reference_length, 0);
    for (std::size_t b = 0; b < requests.size(); ++b) {
        std::copy(
            requests[b]->token_ids().begin(),
            requests[b]->token_ids().end(),
            reference_tokens.begin() +
                static_cast<std::ptrdiff_t>(b * reference_length));
    }

    gpt2_forward(
        &model, reference_tokens.data(), nullptr,
        requests.size(), reference_length);

    for (std::size_t b = 0; b < requests.size(); ++b) {
        const Sequence& sequence = *requests[b];
        for (std::size_t generated_index = sequence.num_prompt_tokens();
             generated_index < sequence.num_tokens(); ++generated_index) {
            const std::size_t logit_position = generated_index - 1;
            const float* logits =
                model.acts.logits +
                (b * reference_length + logit_position) *
                    model.config.padded_vocab_size;
            const int expected =
                reference_argmax(logits, model.config.vocab_size);
            assert(sequence.token_ids()[generated_index] == expected);
        }
    }

    assert(inference_activations < model.num_activations);
    std::cout
        << "GPT2Engine test passed: mixed decode/prefill, dynamic admission, "
           "block reuse, full-prefix greedy agreement\n"
        << "inference_workspace_activations=" << inference_activations
        << " reference_activations=" << model.num_activations << '\n'
        << "request_lengths=" << request1->num_tokens() << ','
        << request2->num_tokens() << ',' << request3->num_tokens()
        << " free_blocks=" << engine.num_free_blocks() << '\n';

    gpt2_free(&model);
    return 0;
}
