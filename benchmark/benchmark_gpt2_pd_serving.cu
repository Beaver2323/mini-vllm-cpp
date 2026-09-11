#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../train_gpt2.cpp"
#include "../mini_vllm/cuda/gpt2_cuda_engine.hpp"
#include "../mini_vllm/cuda/gpt2_pd_engine.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace mini_vllm;
using namespace mini_vllm::cuda;
using Clock = std::chrono::steady_clock;

static std::vector<int> make_prompt(int n, int seed) {
    std::vector<int> p(n); p[0] = 50256;
    for (int i = 1; i < n; ++i) p[i] = 100 + (seed * 3571 + i * 7919) % 50000;
    return p;
}
struct Result {
    double total_ms = 0, transfer_ms = 0;
    std::size_t bytes = 0, concurrent_steps = 0;
    std::vector<std::vector<int>> tokens;
};
static Result run_single(GPT2CudaEngine& engine, std::uint64_t id) {
    std::vector<std::shared_ptr<Sequence>> requests;
    const auto start = Clock::now();
    for (int i = 0; i < 3; ++i)
        requests.push_back(engine.add_request(id + i, make_prompt(17 + 16 * i, i + 1), {8, -1, true}));
    while (!engine.is_finished()) engine.step();
    Result result;
    result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (auto& request : requests)
        result.tokens.emplace_back(request->token_ids().begin() + request->num_prompt_tokens(), request->token_ids().end());
    return result;
}
static Result run_pd(GPT2PDEngine& engine, std::uint64_t id, bool trace) {
    std::vector<std::shared_ptr<PDRequest>> requests;
    Result result;
    std::vector<PDStepResult> steps;
    const auto start = Clock::now();
    for (int i = 0; i < 3; ++i)
        requests.push_back(engine.add_request(id + i, make_prompt(17 + 16 * i, i + 1), {8, -1, true}));
    while (!engine.is_finished()) {
        auto step = engine.step();
        result.concurrent_steps += step.concurrent_submissions;
        steps.push_back(std::move(step));
    }
    result.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (trace) {
        for (std::size_t i = 0; i < steps.size(); ++i) {
            const auto& step = steps[i];
            std::cout << "step=" << i << " GPU0_prefill=" << step.prefill_tokens
                      << " GPU1_decode=" << step.decode_tokens << " handoff=" << step.handed_off.size()
                      << " concurrent=" << step.concurrent_submissions << '\n';
        }
    }
    for (auto& request : requests) {
        auto& s = *request->sequence;
        result.tokens.emplace_back(s.token_ids().begin() + s.num_prompt_tokens(), s.token_ids().end());
        result.bytes += request->transfer.payload_bytes;
        result.transfer_ms += request->transfer.total_ms;
        if (trace) {
            std::cout << "request=" << s.request_id() << " KV_pages=" << request->transfer.num_pages
                      << " payload_bytes=" << request->transfer.payload_bytes << " handoff_ms=" << request->transfer.total_ms
                      << " output=";
            for (int token : result.tokens.back()) std::cout << token << ' ';
            std::cout << '\n';
        }
    }
    return result;
}
int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "benchmark/results/pd_serving_comparison.csv";
        const bool graph = argc > 2 && std::string(argv[2]) == "--cuda-graph";
        GPT2 model{}; gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        GPT2CudaConfig config{model.config.max_seq_len, model.config.vocab_size,
            model.config.padded_vocab_size, model.config.num_layers, model.config.num_heads,
            model.config.channels, CudaDataType::FP16, false, graph, true, 0};
        GPT2CudaEngine single(config, model.params_memory, model.num_parameters, 24, {3, 32}, 64);
        GPT2PDEngine pd(config, model.params_memory, model.num_parameters, 0, 1, 8, 16, 32, 3, 64);
        cudaDeviceProp p{}, d{};
        if (cudaGetDeviceProperties(&p, 0) != cudaSuccess || cudaGetDeviceProperties(&d, 1) != cudaSuccess)
            throw std::runtime_error("cannot query PD GPUs");
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("cannot open PD benchmark CSV");
        csv << "gpu0,gpu1,precision,cuda_graph,mode,repeat,total_ms,output_tokens_per_second,kv_payload_bytes,host_transfer_bytes,transfer_ms,concurrent_steps\n";
        csv << std::fixed << std::setprecision(6);
        for (int repeat = -1; repeat < 7; ++repeat) {
            const auto id = static_cast<std::uint64_t>(repeat + 2) * 100;
            // 交替测量顺序，降低固定顺序带来的偏差。
            Result a, b;
            if (repeat % 2 == 0) { b = run_pd(pd, id, false); a = run_single(single, id); }
            else { a = run_single(single, id); b = run_pd(pd, id, repeat == -1); }
            if (a.tokens != b.tokens) throw std::runtime_error("PD differs from single GPU");
            if (repeat < 0) continue;
            for (int mode = 0; mode < 2; ++mode) {
                const auto& r = mode == 0 ? a : b;
                csv << p.name << ',' << d.name << ",fp16," << (graph ? "true" : "false") << ','
                    << (mode == 0 ? "single" : "pd") << ',' << repeat << ',' << r.total_ms << ','
                    << 24000 / r.total_ms << ',' << r.bytes << ',' << 2 * r.bytes << ','
                    << r.transfer_ms << ',' << r.concurrent_steps << '\n';
            }
        }
        gpt2_free(&model);
        std::cout << "PD/single GPU generated tokens equal; CSV: " << path << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
