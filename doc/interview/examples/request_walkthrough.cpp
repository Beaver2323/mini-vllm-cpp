// CPU 版运行真实控制面、使用教学 Token；WALK_CUDA 版运行真实 CUDA Runner。
#include "mini_vllm/model_input.hpp"
#ifdef WALK_CUDA
#include "mini_vllm/cuda/gpt2_cuda_model_runner.cuh"
#endif
#include <cassert>
#include <iostream>
#include <numeric>
#include <string>

using namespace mini_vllm;

template<class T> void print_vector(const char* name, const std::vector<T>& values) {
    std::cout << name << "=[";
    for (std::size_t i = 0; i < values.size(); ++i)
        std::cout << (i ? "," : "") << values[i];
    std::cout << "]\n";
}

void print_state(const char* stage, const Sequence& s) {
    const char* status = s.is_finished() ? "Finished" :
        (s.status() == SequenceStatus::Waiting ? "Waiting" : "Running");
    std::cout << stage << " request=" << s.request_id() << " status=" << status
              << " computed=" << s.num_computed_tokens() << " total=" << s.num_tokens()
              << " pending=" << s.pending_tokens() << " completion=" << s.num_completion_tokens()
              << " phase_by_counter=" << (s.is_prefill() ? "Prefill" : "Decode") << '\n';
    print_vector("pages", s.block_table());
}

#ifdef WALK_CUDA
// 与 Runner 的 parameter_sizes / point_parameters 顺序一致。无下载、无预训练权重。
std::vector<float> toy_parameters() {
    const std::size_t c = 32, l = 2, vp = 64, t = 32;
    const std::vector<std::size_t> sizes{
        vp*c,t*c,l*c,l*c,l*3*c*c,l*3*c,l*c*c,l*c,
        l*c,l*c,l*4*c*c,l*4*c,l*c*4*c,l*c,c,c};
    std::vector<float> parameters;
    for (std::size_t tensor = 0; tensor < sizes.size(); ++tensor) {
        for (std::size_t i = 0; i < sizes[tensor]; ++i) {
            const bool norm = tensor == 2 || tensor == 8 || tensor == 14;
            const bool weight = tensor == 0 || tensor == 1 || tensor == 4 ||
                                tensor == 6 || tensor == 10 || tensor == 12;
            const int code = static_cast<int>((i*17 + tensor*13) % 101) - 50;
            parameters.push_back(norm ? 1.0f : (weight ? code * 0.001f : 0.0f));
        }
    }
    return parameters;
}
#endif

int main() {
    try {
        BlockManager blocks(4, 16, false);
        Scheduler scheduler({2, 16}, blocks);
        std::vector<int> a_prompt(17), b_prompt(19);
        std::iota(a_prompt.begin(), a_prompt.end(), 1);
        std::iota(b_prompt.begin(), b_prompt.end(), 20);
        auto a = std::make_shared<Sequence>(1, a_prompt, SamplingParams{3, -1, true});
        auto b = std::make_shared<Sequence>(2, b_prompt, SamplingParams{2, -1, true});
        scheduler.add(a); scheduler.add(b);
#ifdef WALK_CUDA
        using namespace mini_vllm::cuda;
        GPT2CudaConfig config{32, 64, 64, 2, 4, 32};
        const auto parameters = toy_parameters();
        GPT2CudaModelRunner runner(config, parameters.data(), parameters.size(), blocks, 2, 16, 32);
        std::cout << "mode=CUDA tiny synthetic GPT-2, FP32, eager, no prefix cache\n";
#else
        std::cout << "mode=CPU control plane, fake samples, NO model or KV execution\n";
#endif
        const std::vector<std::vector<int>> expected_rows{{}, {0}, {0,4}, {0,1}};
        const std::vector<std::size_t> expected_n{16,16,5,2};
        const std::vector<std::size_t> expected_free{3,1,0,0};
        const std::vector<std::size_t> a_computed{16,17,18,19}, b_computed{0,15,19,20};
        int round = 0;
        while (!scheduler.is_finished()) {
            assert(round < 4);
            std::cout << "\nROUND " << round + 1 << '\n';
            print_state("before", *a); print_state("before", *b);
            const auto output = scheduler.schedule();
            const auto input = prepare_packed_model_input(output, blocks, 32, 2, 4);
            std::vector<int> rows;
            for (std::size_t i = 0; i < output.items.size(); ++i) {
                const auto& item = output.items[i];
                if (item.num_scheduled_tokens == item.sequence->pending_tokens())
                    rows.push_back(static_cast<int>(input.query_start_locations[i+1] - 1));
                std::cout << "schedule request=" << item.sequence->request_id()
                          << " phase=" << (item.phase == ExecutionPhase::Prefill ? "Prefill" : "Decode")
                          << " count=" << item.num_scheduled_tokens << '\n';
            }
            assert(rows == expected_rows[round] && input.batch_size() == expected_n[round]);
            assert(blocks.num_free_blocks() == expected_free[round]);
            std::cout << "N=" << input.batch_size() << " R=" << rows.size()
                      << " free_after_schedule=" << blocks.num_free_blocks() << '\n';
            print_state("scheduled", *a); print_state("scheduled", *b);
            print_vector("token_ids", input.token_ids); print_vector("request_ids", input.request_ids);
            print_vector("positions", input.positions); print_vector("context_lengths", input.context_lengths);
            print_vector("slot_mapping", input.slot_mapping); print_vector("block_tables_flat", input.block_tables);
            print_vector("query_start_locations", input.query_start_locations); print_vector("sample_rows", rows);
#ifdef WALK_CUDA
            const auto sampled = runner.run(output);
            assert(runner.last_logit_token_indices() == rows);
            assert(runner.last_model_inputs()[0].slot_mapping == input.slot_mapping);
            std::cout << "metadata_h2d_bytes=" << runner.last_host_to_device_bytes() << '\n';
#else
            std::vector<int> sampled(output.items.size(), -1);
            for (std::size_t i = 0; i < output.items.size(); ++i) {
                const auto& item = output.items[i];
                if (item.num_scheduled_tokens == item.sequence->pending_tokens())
                    sampled[i] = static_cast<int>(item.sequence->request_id() * 10 +
                                                 item.sequence->num_completion_tokens() + 1);
            }
#endif
            print_vector("samples_by_item", sampled);
            // Runner / 教学替身返回后，computed 仍未推进；commit 才更新计数。
            print_state("after_run_before_commit", *a); print_state("after_run_before_commit", *b);
            scheduler.commit(output, sampled);
            assert(a->num_computed_tokens() == a_computed[round]);
            assert(b->num_computed_tokens() == b_computed[round]);
            print_state("committed", *a); print_state("committed", *b);
            std::cout << "free_after_commit=" << blocks.num_free_blocks()
                      << " waiting=" << scheduler.num_waiting() << " running=" << scheduler.num_running() << '\n';
            ++round;
        }
        assert(round == 4 && blocks.num_free_blocks() == 4);
        assert(a->num_tokens() == 20 && b->num_tokens() == 21);
        // 独立容量探针：验证释放后的页号可重用，不向 Scheduler 增加第三个生成请求。
        Sequence probe(3, std::vector<int>(17, 7), {1, -1, true});
        assert(blocks.ensure_capacity(probe, 17));
        assert((probe.block_table() == std::vector<int>{1, 0}));
        print_vector("recycled_pages", probe.block_table());
        blocks.release(probe); blocks.validate();
        std::cout << "PASS: four rounds, mixed batch, page boundary, R=0, release/reuse\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
