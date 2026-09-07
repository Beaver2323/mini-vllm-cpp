#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../train_gpt2.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int B = 2;
    constexpr int T = 33;
    constexpr int second_request_start = 4;
    constexpr int second_request_length = 20;
    std::vector<int> tokens(B * T);
    for (int b = 0; b < B; ++b) {
        tokens[b * T] = 50256;
        for (int i = 1; i < T; ++i) {
            tokens[b * T + i] =
                100 + ((b + 1) * 3571 + i * 7919) % 50000;
        }
    }

    GPT2 reference_model;
    GPT2 incremental_model;
    gpt2_build_from_checkpoint(&reference_model, "gpt2_124M.bin");
    gpt2_build_from_checkpoint(&incremental_model, "gpt2_124M.bin");

    // Allocate each model's activation arena for the maximum batch/sequence shape.
    // The incremental path reuses its arena as a compact active-batch workspace.
    gpt2_forward(&reference_model, tokens.data(), nullptr, B, T);
    gpt2_forward(&incremental_model, tokens.data(), nullptr, B, T);

    const int max_blocks_per_sequence = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    KVCachePool pool(B * max_blocks_per_sequence,
                     incremental_model.config.num_layers,
                     incremental_model.config.num_heads,
                     incremental_model.config.channels /
                         incremental_model.config.num_heads);
    // Force logical blocks to use reversed and interleaved physical ids.
    std::reverse(pool.free_pages.begin(), pool.free_pages.end());
    PageTable table(B, max_blocks_per_sequence);

    const int vocab = reference_model.config.vocab_size;
    const int padded_vocab = reference_model.config.padded_vocab_size;
    double global_max_abs_error = 0.0;
    double global_max_rel_error = 0.0;
    int worst_request = -1;
    int worst_position = -1;
    int worst_vocab_index = -1;

    auto compare_logits = [&](int active_row, int reference_batch,
                              int position) {
        const float* expected = reference_model.acts.logits +
                                (reference_batch * T + position) * padded_vocab;
        const float* actual =
            incremental_model.acts.logits + active_row * padded_vocab;
        double position_max_abs_error = 0.0;
        for (int v = 0; v < vocab; ++v) {
            const double abs_error = std::abs(double(actual[v]) - expected[v]);
            const double rel_error =
                abs_error / std::max(1e-6, std::abs(double(expected[v])));
            position_max_abs_error = std::max(position_max_abs_error, abs_error);
            global_max_rel_error = std::max(global_max_rel_error, rel_error);
            if (abs_error > global_max_abs_error) {
                global_max_abs_error = abs_error;
                worst_request = reference_batch;
                worst_position = position;
                worst_vocab_index = v;
            }
        }
        return position_max_abs_error;
    };

    int second_position = 0;
    for (int first_position = 0; first_position < T; ++first_position) {
        if (first_position % PAGE_SIZE == 0) {
            table.block_tables[first_position / PAGE_SIZE] = pool.allocate_page();
        }
        table.context_lengths[0] = first_position + 1;
        int current_tokens[B] = {tokens[first_position], 0};
        int active_batch_size = 1;

        const bool second_is_active =
            first_position >= second_request_start &&
            second_position < second_request_length;
        if (second_is_active) {
            if (second_position % PAGE_SIZE == 0) {
                table.block_tables[max_blocks_per_sequence +
                                   second_position / PAGE_SIZE] =
                    pool.allocate_page();
            }
            table.context_lengths[1] = second_position + 1;
            current_tokens[1] = tokens[T + second_position];
            active_batch_size = 2;
        }

        gpt2_forward_inference_batched(&incremental_model, current_tokens, &pool,
                                       &table, active_batch_size);
        const double first_error = compare_logits(0, 0, first_position);
        if (first_position == 0 || first_position == 14 ||
            first_position == 15 || first_position == 16 ||
            first_position == 30 || first_position == 31 ||
            first_position == 32) {
            std::printf("request=0 length=%d max_abs_error=%.9g\n",
                        first_position + 1, first_error);
        }
        if (second_is_active) {
            const double second_error = compare_logits(1, 1, second_position);
            if (second_position == 0 || second_position == 14 ||
                second_position == 15 || second_position == 16 ||
                second_position == second_request_length - 1) {
                std::printf("request=1 length=%d max_abs_error=%.9g\n",
                            second_position + 1, second_error);
            }
            ++second_position;
        }
    }

    std::printf("request0_lengths=1..%d request1_lengths=1..%d "
                "request1_join_step=%d max_abs_error=%.9g max_rel_error=%.9g "
                "worst_request=%d worst_position=%d worst_vocab_index=%d\n",
                T, second_request_length, second_request_start + 1,
                global_max_abs_error, global_max_rel_error, worst_request,
                worst_position, worst_vocab_index);

    gpt2_free(&reference_model);
    gpt2_free(&incremental_model);
    return global_max_abs_error < 2e-3 ? 0 : 1;
}
