#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../../train_gpt2.cpp"
#include "../../mini_vllm/cuda/gpt2_cuda_model_runner.cuh"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace mini_vllm;
using namespace mini_vllm::cuda;

int main(int argc, char** argv) {
    try {
        GPT2 model{}; gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        const auto dtype = argc > 1 && std::string(argv[1]) == "--fp32" ? CudaDataType::FP32 : CudaDataType::FP16;
        GPT2CudaConfig config{model.config.max_seq_len, model.config.vocab_size,
            model.config.padded_vocab_size, model.config.num_layers, model.config.num_heads,
            model.config.channels, dtype, false, true, true};
        BlockManager pruned_blocks(8, 16), full_blocks(8, 16);
        GPT2CudaModelRunner pruned(config, model.params_memory, model.num_parameters, pruned_blocks, 2, 4, 16);
        config.enable_sample_row_pruning = false;
        config.enable_cuda_graph = false;
        GPT2CudaModelRunner full(config, model.params_memory, model.num_parameters, full_blocks, 2, 4, 16);
        const std::vector<std::vector<int>> lengths{{8}, {4}, {2, 2}, {2, 8}, {8}};
        const std::vector<std::vector<int>> expected_rows{{}, {3}, {1, 3}, {1}, {}};
        double max_error = 0;
        for (std::size_t test = 0; test < lengths.size(); ++test) {
            SchedulerOutput a, b;
            for (std::size_t i = 0; i < lengths[test].size(); ++i) {
                std::vector<int> tokens(lengths[test][i]);
                tokens[0] = 50256;
                for (std::size_t j = 1; j < tokens.size(); ++j) tokens[j] = 100 + (j * 7919 + test * 3571) % 50000;
                auto sa = std::make_shared<Sequence>(test * 10 + i, tokens, SamplingParams{1, -1, true});
                auto sb = std::make_shared<Sequence>(test * 10 + i, tokens, SamplingParams{1, -1, true});
                assert(pruned_blocks.ensure_capacity(*sa, tokens.size()));
                assert(full_blocks.ensure_capacity(*sb, tokens.size()));
                const std::size_t n = lengths[test].size() == 1 ? 4 : 2;
                a.items.push_back({sa, ExecutionPhase::Prefill, n});
                b.items.push_back({sb, ExecutionPhase::Prefill, n});
                a.num_batched_tokens += n; b.num_batched_tokens += n;
            }
            const auto samples_a = pruned.run(a), samples_b = full.run(b);
            assert(samples_a == samples_b);
            assert(pruned.last_logit_token_indices() == expected_rows[test]);
            assert(pruned.num_cuda_graphs() == std::min(test + 1, std::size_t{3}));
            const auto la = pruned.last_logits_for_testing(), lb = full.last_logits_for_testing();
            assert(la.size() == expected_rows[test].size() * model.config.padded_vocab_size);
            assert(lb.size() == 4ul * model.config.padded_vocab_size);
            for (std::size_t row = 0; row < expected_rows[test].size(); ++row) {
                for (int v = 0; v < model.config.vocab_size; ++v) {
                    const double error = std::abs(double(la[row * model.config.padded_vocab_size + v]) -
                        lb[expected_rows[test][row] * model.config.padded_vocab_size + v]);
                    max_error = std::max(max_error, error);
                }
            }
            for (auto& item : a.items) pruned_blocks.release(*item.sequence);
            for (auto& item : b.items) full_blocks.release(*item.sequence);
        }
        assert(max_error < 0.005);
        std::cout << "sample rows [0,1,2,1,0], same N/different R, dynamic gather indices, graph replay PASS; max_error="
                  << max_error << '\n';
        gpt2_free(&model);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
