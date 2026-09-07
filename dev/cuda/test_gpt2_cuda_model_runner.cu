#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../../train_gpt2.cpp"
#include "../../mini_vllm/cuda/gpt2_cuda_model_runner.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace mini_vllm;
using mini_vllm::cuda::GPT2CudaConfig;
using mini_vllm::cuda::GPT2CudaModelRunner;

static std::vector<int> make_prompt(std::size_t length, int seed) {
    std::vector<int> tokens(length);
    tokens[0] = 50256;
    for (std::size_t index = 1; index < length; ++index) {
        tokens[index] =
            100 + (seed * 3571 + static_cast<int>(index) * 7919) % 50000;
    }
    return tokens;
}

static int argmax(const float* logits, int vocab_size) {
    return static_cast<int>(
        std::max_element(logits, logits + vocab_size) - logits);
}

struct StepValidation {
    double max_abs_logit_error = 0.0;
    std::size_t expected_metadata_bytes = 0;
};

static StepValidation validate_and_commit(
    GPT2& model, GPT2DenseInferenceWorkspace& reference_workspace,
    Scheduler& scheduler, GPT2CudaModelRunner& runner,
    const SchedulerOutput& output) {
    const std::vector<int> sampled = runner.run(output);
    const std::vector<ModelInput>& inputs = runner.last_model_inputs();
    assert(inputs.size() == 1);
    assert(inputs.front().batch_size() == output.num_batched_tokens);
    assert(inputs.front().query_start_locations.size() ==
           output.items.size() + 1);

    StepValidation validation;
    for (const ModelInput& input : inputs) {
        validation.expected_metadata_bytes += sizeof(int) *
            (4 * input.batch_size() + input.block_tables.size());
    }
    assert(runner.last_host_to_device_bytes() ==
           validation.expected_metadata_bytes);

    const ModelInput& final_input = inputs.back();
    const std::vector<float> gpu_logits =
        runner.last_logits_for_testing();
    assert(gpu_logits.size() ==
           final_input.batch_size() *
               static_cast<std::size_t>(model.config.padded_vocab_size));

    for (std::size_t row = 0; row < final_input.batch_size(); ++row) {
        const std::size_t item_index =
            final_input.scheduled_item_indices[row];
        const Sequence& sequence = *output.items[item_index].sequence;
        const int prefix_length = final_input.positions[row] + 1;
        gpt2_forward_dense_with_workspace(
            &model, sequence.token_ids().data(), 1, prefix_length,
            &reference_workspace);
        const float* expected =
            reference_workspace.acts().logits +
            static_cast<std::size_t>(prefix_length - 1) *
                model.config.padded_vocab_size;
        const float* actual =
            gpu_logits.data() + row * model.config.padded_vocab_size;
        for (int token_id = 0; token_id < model.config.vocab_size;
             ++token_id) {
            validation.max_abs_logit_error = std::max(
                validation.max_abs_logit_error,
                std::abs(static_cast<double>(actual[token_id]) -
                         expected[token_id]));
        }
        assert(argmax(actual, model.config.vocab_size) ==
               argmax(expected, model.config.vocab_size));
    }

    for (std::size_t item_index = 0; item_index < output.items.size();
         ++item_index) {
        if (sampled[item_index] < 0) continue;
        const Sequence& sequence = *output.items[item_index].sequence;
        gpt2_forward_dense_with_workspace(
            &model, sequence.token_ids().data(), 1,
            static_cast<int>(sequence.num_tokens()),
            &reference_workspace);
        const float* expected =
            reference_workspace.acts().logits +
            (sequence.num_tokens() - 1) *
                model.config.padded_vocab_size;
        assert(sampled[item_index] ==
               argmax(expected, model.config.vocab_size));
    }

    scheduler.commit(output, sampled);
    return validation;
}

int main() {
    GPT2 model{};
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    constexpr std::size_t max_context_length = 64;
    constexpr std::size_t max_num_sequences = 2;
    // 前两个并发请求恰好占满 3 页，释放后第三个请求必须复用旧页。
    BlockManager block_manager(/*num_blocks=*/3, PAGE_SIZE);
    Scheduler scheduler(
        {/*max_num_sequences=*/max_num_sequences,
         /*max_num_batched_tokens=*/8},
        block_manager);
    const GPT2CudaConfig cuda_config{
        model.config.max_seq_len,
        model.config.vocab_size,
        model.config.padded_vocab_size,
        model.config.num_layers,
        model.config.num_heads,
        model.config.channels,
    };
    GPT2CudaModelRunner runner(
        cuda_config, model.params_memory, model.num_parameters,
        block_manager, max_num_sequences,
        /*max_num_batched_tokens=*/8, max_context_length);
    GPT2DenseInferenceWorkspace reference_workspace(
        model.config, 1, max_context_length);

    auto request1 = std::make_shared<Sequence>(
        1, make_prompt(17, 1),
        SamplingParams{/*max_new_tokens=*/2, -1, false});
    auto request2 = std::make_shared<Sequence>(
        2, make_prompt(5, 2),
        SamplingParams{/*max_new_tokens=*/1, -1, false});
    auto request3 = std::make_shared<Sequence>(
        3, make_prompt(17, 3),
        SamplingParams{/*max_new_tokens=*/1, -1, false});

    scheduler.add(request1);
    double global_max_abs_logit_error = 0.0;
    std::size_t total_metadata_bytes = 0;
    bool saw_mixed_batch = false;
    bool saw_reused_block = false;
    int step = 0;
    while (!scheduler.is_finished()) {
        ++step;
        if (step == 2) scheduler.add(request2);
        if (step == 4) scheduler.add(request3);

        SchedulerOutput output = scheduler.schedule();
        assert(!output.items.empty());
        if (output.items.size() == 2 &&
            output.items[0].phase != output.items[1].phase) {
            saw_mixed_batch = true;
        }
        for (const ScheduledItem& item : output.items) {
            if (item.sequence->request_id() == 3) {
                for (int block_id : item.sequence->block_table()) {
                    if (block_id == 0 || block_id == 1 || block_id == 2) {
                        saw_reused_block = true;
                    }
                }
            }
        }

        const StepValidation validation = validate_and_commit(
            model, reference_workspace, scheduler, runner, output);
        global_max_abs_logit_error = std::max(
            global_max_abs_logit_error,
            validation.max_abs_logit_error);
        total_metadata_bytes += validation.expected_metadata_bytes;
    }

    assert(request1->is_finished());
    assert(request2->is_finished());
    assert(request3->is_finished());
    assert(saw_mixed_batch);
    assert(saw_reused_block);
    assert(block_manager.num_free_blocks() == block_manager.num_blocks());
    assert(global_max_abs_logit_error < 0.2);

    std::cout
        << "CUDA GPT2ModelRunner test passed: full GPU decode, mixed batch, "
           "cross-page growth, block reuse, CPU greedy agreement\n"
        << "max_abs_logit_error=" << global_max_abs_logit_error << '\n'
        << "weight_bytes=" << runner.weight_bytes()
        << " kv_cache_bytes=" << runner.kv_cache_bytes()
        << " activation_bytes=" << runner.activation_bytes()
        << " metadata_h2d_bytes=" << total_metadata_bytes << '\n'
        << "request_lengths=" << request1->num_tokens() << ','
        << request2->num_tokens() << ',' << request3->num_tokens()
        << " free_blocks=" << block_manager.num_free_blocks() << '\n';

    gpt2_free(&model);
    return 0;
}
