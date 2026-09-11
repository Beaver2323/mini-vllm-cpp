#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../../train_gpt2.cpp"
#include "../../mini_vllm/cuda/gpt2_cuda_engine.hpp"
#include "../../mini_vllm/cuda/gpt2_pd_engine.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <iostream>

using namespace mini_vllm;
using namespace mini_vllm::cuda;

static std::vector<int> prompt(int n, int seed = 1) {
    std::vector<int> tokens(n); tokens[0] = 50256;
    for (int i = 1; i < n; ++i) tokens[i] = 100 + (seed * 3571 + i * 7919) % 50000;
    return tokens;
}
static std::vector<int> completion(const Sequence& s) {
    return {s.token_ids().begin() + s.num_prompt_tokens(), s.token_ids().end()};
}
template<class F> static void must_throw(F&& f) {
    bool threw = false; try { f(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

// 非相同物理页号 + 全词表比较：避免仅比较 Greedy 恰巧相等而漏掉迁移错误。
static void test_remapped_transfer(GPT2& model, GPT2CudaConfig config) {
    BlockManager src_blocks(6, 16), dst_blocks(6, 16);
    config.device_id = 0;
    GPT2CudaModelRunner src(config, model.params_memory, model.num_parameters, src_blocks, 1, 64, 64);
    config.device_id = 1;
    GPT2CudaModelRunner dst(config, model.params_memory, model.num_parameters, dst_blocks, 1, 64, 64);
    auto source = std::make_shared<Sequence>(1, prompt(17), SamplingParams{4, -1, true});
    auto target = std::make_shared<Sequence>(1, prompt(17), SamplingParams{4, -1, true});
    Sequence blocker(99, prompt(1), {1, -1, true});
    assert(src_blocks.ensure_capacity(*source, 20));
    assert(dst_blocks.ensure_capacity(blocker, 16));
    assert(dst_blocks.ensure_capacity(*target, 20));
    assert(source->block_table()[0] != target->block_table()[0]);
    SchedulerOutput prefill{{{source, ExecutionPhase::Prefill, 17}}, 17};
    const auto first = src.run(prefill)[0]; source->mark_computed(17); source->append_token(first);
    const auto stats = src.copy_kv_to(dst, *source, *target, 17);
    assert(stats.num_pages == 2 && stats.payload_bytes == 2 * 2 * 16 * 12 * 768 * 2);
    assert(src_blocks.num_free_blocks() == 4); // copy 不负责回收源页。
    target->mark_computed(17); target->append_token(first);
    for (int i = 0; i < 3; ++i) {
        SchedulerOutput a{{{source, ExecutionPhase::Decode, 1}}, 1};
        SchedulerOutput b{{{target, ExecutionPhase::Decode, 1}}, 1};
        const int sa = src.run(a)[0], sb = dst.run(b)[0]; assert(sa == sb);
        const auto la = src.last_logits_for_testing(), lb = dst.last_logits_for_testing();
        assert(la.size() == lb.size());
        double error = 0;
        for (int v = 0; v < model.config.vocab_size; ++v)
            error = std::max(error, std::abs(double(la[v]) - lb[v]));
        assert(error < 1e-5);
        source->mark_computed(1); source->append_token(sa);
        target->mark_computed(1); target->append_token(sb);
    }
    must_throw([&]{ src.copy_kv_to(src, *source, *target, 17); });
    must_throw([&]{ src.copy_kv_to(dst, *source, *target, 64); });
    Sequence wrong(2, prompt(17, 7), {1, -1, true});
    must_throw([&]{ src.copy_kv_to(dst, *source, wrong, 17); });
    src_blocks.release(*source); dst_blocks.release(*target); dst_blocks.release(blocker);
    assert(src_blocks.num_free_blocks() == 6 && dst_blocks.num_free_blocks() == 6);
    int current = -1; assert(cudaGetDevice(&current) == cudaSuccess); assert(current == 0);
    std::cout << "remapped KV transfer + full-vocabulary decode logits PASS\n";
}

static void test_pipeline(GPT2& model, GPT2CudaConfig config, bool constrained) {
    const std::vector<int> lengths = {1, 16, 17, 31, 32, 33};
    const std::size_t d_blocks = constrained ? 3 : 16;
    GPT2PDEngine pd(config, model.params_memory, model.num_parameters, 0, 1,
                    4, d_blocks, 16, 3, 64);
    config.device_id = 0;
    GPT2CudaEngine single(config, model.params_memory, model.num_parameters, 24, {3, 16}, 64);
    std::vector<std::shared_ptr<PDRequest>> requests;
    std::vector<std::shared_ptr<Sequence>> references;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        const SamplingParams sampling{i == 0 ? 1ul : 6ul, -1, true};
        requests.push_back(pd.add_request(i, prompt(lengths[i], i + 1), sampling));
        references.push_back(single.add_request(i, prompt(lengths[i], i + 1), sampling));
    }
    must_throw([&]{ pd.add_request(0, prompt(1), {1, -1, true}); });
    must_throw([&]{ pd.add_request(100, prompt(64), {2, -1, true}); });
    must_throw([&]{ pd.add_request(101, {-1}, {1, -1, true}); });
    std::size_t transfers = 0, concurrent = 0, iterations = 0;
    while (!pd.is_finished()) {
        assert(++iterations < 100);
        auto step = pd.step();
        transfers += step.handed_off.size(); concurrent += step.concurrent_submissions;
        assert(step.sampled_request_ids.size() == step.sampled_token_ids.size());
    }
    while (!single.is_finished()) single.step();
    assert(transfers == lengths.size() - 1 && concurrent > 0);
    GPT2DenseInferenceWorkspace workspace(model.config, 1, 64);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = *requests[i];
        assert(request.stage == PDStage::Finished && request.sequence->is_finished());
        assert(request.sequence->block_table().empty());
        assert(completion(*request.sequence) == completion(*references[i]));
        auto tokens = prompt(lengths[i], i + 1);
        for (int expected : completion(*request.sequence)) {
            gpt2_forward_dense_with_workspace(&model, tokens.data(), 1, tokens.size(), &workspace);
            const float* logits = workspace.acts().logits + (tokens.size() - 1) * model.config.padded_vocab_size;
            assert(expected == std::max_element(logits, logits + model.config.vocab_size) - logits);
            tokens.push_back(expected);
        }
    }
    assert(requests[0]->transfer.payload_bytes == 0);
    assert(pd.prefill_free_blocks() == 4 && pd.decode_free_blocks() == d_blocks);
    // EOS 出现在 P 首 Token 时应直接完成，不交接。
    const int eos = requests[0]->sequence->token_ids().back();
    auto early = pd.add_request(200, prompt(1), {6, eos, false});
    while (!pd.is_finished()) pd.step();
    assert(early->sequence->num_completion_tokens() == 1 && early->transfer.payload_bytes == 0);
    std::cout << "PD pipeline constrained=" << constrained << " transfers=" << transfers
              << " concurrent_steps=" << concurrent << " CPU/single-GPU equality and reclamation PASS\n";
}

int main(int argc, char** argv) {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
            throw std::runtime_error("PD tests require two CUDA GPUs");
        assert(cudaSetDevice(0) == cudaSuccess);
        GPT2 model{}; gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        GPT2CudaConfig config{model.config.max_seq_len, model.config.vocab_size,
            model.config.padded_vocab_size, model.config.num_layers, model.config.num_heads,
            model.config.channels, CudaDataType::FP16, false, argc > 1 && std::string(argv[1]) == "--cuda-graph", true};
        test_remapped_transfer(model, config);
        test_pipeline(model, config, false);
        test_pipeline(model, config, true);
        gpt2_free(&model);
        std::cout << "All PD tests PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
