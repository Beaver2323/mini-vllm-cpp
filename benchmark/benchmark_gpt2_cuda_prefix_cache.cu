#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../train_gpt2.cpp"
#include "../mini_vllm/cuda/gpt2_cuda_engine.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace mini_vllm;
using namespace mini_vllm::cuda;
using Clock = std::chrono::steady_clock;

struct Observation {
    double total_ms = 0, ttft_ms = 0;
    std::size_t scheduled_tokens = 0, hit_blocks = 0;
    std::vector<int> generated;
};

Observation measure(GPT2CudaEngine& engine, std::uint64_t id, const std::vector<int>& prompt,
                    std::size_t new_tokens = 4) {
    const auto hit_before = engine.prefix_cache_hit_blocks();
    const auto start = Clock::now();
    auto request = engine.add_request(id, prompt, {new_tokens, -1, true});
    Observation result;
    while (!engine.is_finished()) {
        auto step = engine.step();
        result.scheduled_tokens += step.num_batched_tokens;
        if (request->num_completion_tokens() == 1)
            result.ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    result.hit_blocks = engine.prefix_cache_hit_blocks() - hit_before;
    result.generated.assign(request->token_ids().begin() + prompt.size(), request->token_ids().end());
    return result;
}

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "benchmark/results/prefix_cache_comparison.csv";
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 7;
        if (repeats < 1) throw std::invalid_argument("repeats must be positive");
        GPT2 model{};
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        GPT2CudaConfig config{model.config.max_seq_len, model.config.vocab_size,
            model.config.padded_vocab_size, model.config.num_layers, model.config.num_heads,
            model.config.channels, CudaDataType::FP16, false, false, true};
        GPT2CudaEngine off(config, model.params_memory, model.num_parameters, 40, {1, 272}, 272, false);
        GPT2CudaEngine cached(config, model.params_memory, model.num_parameters, 40, {1, 272}, 272, true);
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, off.model_runner().device_id()) != cudaSuccess)
            throw std::runtime_error("cannot query GPU");
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("cannot open prefix benchmark CSV");
        csv << "gpu,precision,fusion,cuda_graph,sample_row_pruning,prefix_tokens,prompt_tokens,mode,repeat,scheduled_tokens,hit_blocks,ttft_ms,total_ms,tpot_ms,output_tokens_per_second\n";
        csv << std::fixed << std::setprecision(6);
        std::uint64_t id = 1;
        GPT2DenseInferenceWorkspace reference(model.config, 1, 272);
        for (int prefix : {16, 64, 128, 256}) {
            std::vector<int> seed(prefix + 1);
            seed[0] = 50256;
            for (int i = 1; i <= prefix; ++i) seed[i] = 100 + (i * 7919) % 50000;
            std::vector<int> prompt(seed.begin(), seed.begin() + prefix);
            prompt.push_back(1234); prompt.push_back(4321);
            // 完整 CPU Greedy 独立参考，计时之外。
            auto tokens = prompt;
            std::vector<int> expected;
            for (int i = 0; i < 4; ++i) {
                gpt2_forward_dense_with_workspace(&model, tokens.data(), 1, tokens.size(), &reference);
                const auto* logits = reference.acts().logits + (tokens.size() - 1) * model.config.padded_vocab_size;
                const int token = std::max_element(logits, logits + model.config.vocab_size) - logits;
                expected.push_back(token); tokens.push_back(token);
            }
            // 每种模式一次预热，再轮换模式测量；所有 seed 和清理均在计时区间之外。
            for (int repeat = -1; repeat < repeats; ++repeat) {
                for (int mode = 0; mode < 3; ++mode) {
                    auto& engine = mode == 0 ? off : cached;
                    engine.clear_prefix_cache();
                    if (mode == 2) measure(engine, id++, seed, 1);
                    auto result = measure(engine, id++, prompt);
                    if (result.generated != expected) throw std::runtime_error("prefix benchmark CPU token mismatch");
                    const auto expected_hits = mode == 2 ? prefix / 16 : 0;
                    const auto expected_scheduled = prompt.size() + 3 - expected_hits * 16;
                    if (result.hit_blocks != static_cast<std::size_t>(expected_hits) ||
                        result.scheduled_tokens != expected_scheduled)
                        throw std::runtime_error("prefix benchmark unexpected cache reuse");
                    if (repeat < 0) continue;
                    const char* name = mode == 0 ? "off" : (mode == 1 ? "miss" : "hit");
                    csv << props.name << ",fp16,false,false,true," << prefix << ',' << prompt.size()
                        << ',' << name << ',' << repeat << ',' << result.scheduled_tokens << ','
                        << result.hit_blocks << ',' << result.ttft_ms << ',' << result.total_ms << ','
                        << (result.total_ms - result.ttft_ms) / 3 << ',' << 4000 / result.total_ms << '\n';
                }
            }
            std::cout << "prefix=" << prefix << " off/miss/hit CPU equality PASS\n";
        }
        gpt2_free(&model);
        std::cout << "Prefix Cache benchmark: " << path << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
