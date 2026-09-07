#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../../train_gpt2.cpp"
#include "../../mini_vllm/cuda/gpt2_cuda_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using mini_vllm::SamplingParams;
using mini_vllm::cuda::CudaDataType;
using mini_vllm::cuda::CudaEngineStepResult;
using mini_vllm::cuda::GPT2CudaConfig;
using mini_vllm::cuda::GPT2CudaEngine;

static int argmax(const float* logits, int vocab_size) {
    return static_cast<int>(
        std::max_element(logits, logits + vocab_size) - logits);
}

static int cpu_greedy(
    GPT2& model, GPT2DenseInferenceWorkspace& workspace,
    const std::vector<int>& prompt) {
    gpt2_forward_dense_with_workspace(
        &model, prompt.data(), 1, static_cast<int>(prompt.size()),
        &workspace);
    const float* logits = workspace.acts().logits +
        (prompt.size() - 1) * model.config.padded_vocab_size;
    return argmax(logits, model.config.vocab_size);
}

int main() {
    GPT2 model{};
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    const GPT2CudaConfig config{
        model.config.max_seq_len,
        model.config.vocab_size,
        model.config.padded_vocab_size,
        model.config.num_layers,
        model.config.num_heads,
        model.config.channels,
        CudaDataType::FP16,
        /*enable_fused_residual_layernorm=*/false,
        /*enable_cuda_graph=*/false,
    };
    GPT2CudaEngine engine(
        config, model.params_memory, model.num_parameters,
        /*num_kv_blocks=*/4,
        {/*max_num_sequences=*/1, /*max_num_batched_tokens=*/64},
        /*max_context_length=*/64,
        /*enable_prefix_cache=*/true);
    GPT2DenseInferenceWorkspace reference(model.config, 1, 64);

    std::vector<int> first_prompt(17);
    first_prompt[0] = 50256;
    for (std::size_t index = 1; index < first_prompt.size(); ++index) {
        first_prompt[index] =
            100 + static_cast<int>((index * 7919) % 50000);
    }
    std::vector<int> second_prompt(
        first_prompt.begin(), first_prompt.begin() + 16);
    second_prompt.push_back(1234);
    second_prompt.push_back(4321);

    const int expected_first = cpu_greedy(model, reference, first_prompt);
    const int expected_second = cpu_greedy(model, reference, second_prompt);

    auto first = engine.add_request(
        1, first_prompt, SamplingParams{1, -1, false});
    const CudaEngineStepResult first_step = engine.step();
    assert(first_step.num_batched_tokens == first_prompt.size());
    assert(first->is_finished());
    assert(first->token_ids().back() == expected_first);
    assert(engine.num_cached_blocks() == 1);

    auto second = engine.add_request(
        2, second_prompt, SamplingParams{1, -1, false});
    const CudaEngineStepResult second_step = engine.step();
    assert(second_step.num_batched_tokens == 2);
    assert(second_step.scheduled_token_counts.size() == 1);
    assert(second_step.scheduled_token_counts[0] == 2);
    assert(second->is_finished());
    assert(second->token_ids().back() == expected_second);
    assert(engine.prefix_cache_hit_blocks() == 1);

    const std::vector<float> gpu_logits =
        engine.model_runner().last_logits_for_testing();
    gpt2_forward_dense_with_workspace(
        &model, second_prompt.data(), 1,
        static_cast<int>(second_prompt.size()), &reference);
    const float* expected_logits = reference.acts().logits +
        (second_prompt.size() - 1) * model.config.padded_vocab_size;
    const float* actual_logits = gpu_logits.data() +
        model.config.padded_vocab_size;
    double max_abs_error = 0.0;
    for (int token = 0; token < model.config.vocab_size; ++token) {
        max_abs_error = std::max(
            max_abs_error,
            std::abs(static_cast<double>(actual_logits[token]) -
                     expected_logits[token]));
    }
    assert(max_abs_error < 0.2);

    std::cout
        << "CUDA prefix cache test passed: cached_blocks="
        << engine.num_cached_blocks()
        << " hit_blocks=" << engine.prefix_cache_hit_blocks()
        << " second_prompt_tokens=" << second_prompt.size()
        << " second_scheduled_tokens=" << second_step.num_batched_tokens
        << " max_abs_logit_error=" << max_abs_error << '\n';

    gpt2_free(&model);
    return 0;
}
