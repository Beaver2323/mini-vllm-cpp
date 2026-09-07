#define GPT2_PAGED_INFERENCE_NO_MAIN
#include "../train_gpt2.cpp"
#include "../mini_vllm/cuda/gpt2_cuda_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MINI_VLLM_GIT_COMMIT
#define MINI_VLLM_GIT_COMMIT "unknown"
#endif

#ifndef MINI_VLLM_GPU_ARCH
#define MINI_VLLM_GPU_ARCH "unknown"
#endif

using mini_vllm::SamplingParams;
using mini_vllm::Sequence;
using mini_vllm::cuda::CudaEngineStepResult;
using mini_vllm::cuda::CudaDataType;
using mini_vllm::cuda::GPT2CudaConfig;
using mini_vllm::cuda::GPT2CudaEngine;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int repeats = 3;
    std::size_t token_budget = 64;
    CudaDataType data_type = CudaDataType::FP32;
    std::string json_path =
        "benchmark/results/gpt2_cuda_packed_rtx3090.json";
    std::string csv_path =
        "benchmark/results/gpt2_cuda_packed_rtx3090.csv";
};

struct RunResult {
    double total_ms = 0.0;
    double throughput_tokens_per_second = 0.0;
    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> request_latency_ms;
    std::vector<std::vector<int>> generated_tokens;
    std::size_t metadata_h2d_bytes = 0;
    std::size_t weight_bytes = 0;
    std::size_t kv_cache_bytes = 0;
    std::size_t activation_bytes = 0;
};

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

std::vector<int> make_prompt(std::size_t length, int seed) {
    std::vector<int> tokens(length);
    tokens[0] = 50256;
    for (std::size_t index = 1; index < length; ++index) {
        tokens[index] =
            100 + (seed * 3571 + static_cast<int>(index) * 7919) % 50000;
    }
    return tokens;
}

int greedy_argmax(const float* logits, int vocab_size) {
    return static_cast<int>(
        std::max_element(logits, logits + vocab_size) - logits);
}

std::vector<std::vector<int>> cpu_greedy_reference(GPT2& model) {
    constexpr std::size_t max_new_tokens = 4;
    const std::vector<std::size_t> prompt_lengths = {8, 16, 24, 32};
    GPT2DenseInferenceWorkspace workspace(model.config, 1, 64);
    std::vector<std::vector<int>> outputs;
    for (std::size_t index = 0; index < prompt_lengths.size(); ++index) {
        std::vector<int> tokens = make_prompt(prompt_lengths[index], index + 1);
        std::vector<int> generated;
        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            gpt2_forward_dense_with_workspace(
                &model, tokens.data(), 1,
                static_cast<int>(tokens.size()), &workspace);
            const float* logits =
                workspace.acts().logits +
                (tokens.size() - 1) * model.config.padded_vocab_size;
            const int sampled = greedy_argmax(logits, model.config.vocab_size);
            generated.push_back(sampled);
            tokens.push_back(sampled);
        }
        outputs.push_back(std::move(generated));
    }
    return outputs;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * (values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double weight = position - lower;
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

RunResult run_once(
    GPT2& model, std::size_t token_budget, CudaDataType data_type) {
    constexpr std::size_t max_num_sequences = 4;
    constexpr std::size_t max_context_length = 64;
    constexpr std::size_t max_new_tokens = 4;
    const GPT2CudaConfig config{
        model.config.max_seq_len,
        model.config.vocab_size,
        model.config.padded_vocab_size,
        model.config.num_layers,
        model.config.num_heads,
        model.config.channels,
        data_type,
    };
    GPT2CudaEngine engine(
        config, model.params_memory, model.num_parameters,
        /*num_kv_blocks=*/16,
        {/*max_num_sequences=*/max_num_sequences,
         /*max_num_batched_tokens=*/token_budget},
        max_context_length);

    const std::vector<std::size_t> prompt_lengths = {8, 16, 24, 32};
    std::vector<std::shared_ptr<Sequence>> requests;
    for (std::size_t index = 0; index < prompt_lengths.size(); ++index) {
        requests.push_back(engine.add_request(
            index + 1, make_prompt(prompt_lengths[index], index + 1),
            SamplingParams{max_new_tokens, -1, false}));
    }

    std::vector<std::vector<double>> token_times(requests.size());
    RunResult result;
    const Clock::time_point start = Clock::now();
    while (!engine.is_finished()) {
        const CudaEngineStepResult step = engine.step();
        const Clock::time_point now = Clock::now();
        result.metadata_h2d_bytes +=
            engine.model_runner().last_host_to_device_bytes();
        for (std::size_t index = 0; index < step.request_ids.size(); ++index) {
            if (step.sampled_token_ids[index] < 0) continue;
            const std::size_t request_index =
                static_cast<std::size_t>(step.request_ids[index] - 1);
            token_times.at(request_index).push_back(
                elapsed_ms(start, now));
        }
    }
    result.total_ms = elapsed_ms(start, Clock::now());
    const std::size_t total_output_tokens =
        requests.size() * max_new_tokens;
    result.throughput_tokens_per_second =
        total_output_tokens * 1000.0 / result.total_ms;

    for (std::size_t index = 0; index < requests.size(); ++index) {
        if (token_times[index].size() != max_new_tokens) {
            throw std::logic_error("CUDA benchmark missed output timestamps");
        }
        result.ttft_ms.push_back(token_times[index].front());
        result.request_latency_ms.push_back(token_times[index].back());
        for (std::size_t token = 1; token < token_times[index].size();
             ++token) {
            result.tpot_ms.push_back(
                token_times[index][token] -
                token_times[index][token - 1]);
        }
        result.generated_tokens.emplace_back(
            requests[index]->token_ids().begin() +
                static_cast<std::ptrdiff_t>(
                    requests[index]->num_prompt_tokens()),
            requests[index]->token_ids().end());
    }
    result.weight_bytes = engine.model_runner().weight_bytes();
    result.kv_cache_bytes = engine.model_runner().kv_cache_bytes();
    result.activation_bytes = engine.model_runner().activation_bytes();
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            return argv[++index];
        };
        if (argument == "--repeats") {
            options.repeats = std::stoi(value());
        } else if (argument == "--token-budget") {
            options.token_budget = std::stoul(value());
        } else if (argument == "--precision") {
            const std::string precision = value();
            if (precision == "fp16") {
                options.data_type = CudaDataType::FP16;
            } else if (precision == "bf16") {
                options.data_type = CudaDataType::BF16;
            } else if (precision != "fp32") {
                throw std::invalid_argument(
                    "precision must be fp32, fp16, or bf16");
            }
        } else if (argument == "--json") {
            options.json_path = value();
        } else if (argument == "--csv") {
            options.csv_path = value();
        } else {
            throw std::invalid_argument(
                "unknown CUDA serving benchmark option: " + argument);
        }
    }
    if (options.repeats <= 0 || options.token_budget == 0) {
        throw std::invalid_argument(
            "repeats and token budget must be positive");
    }
    return options;
}

void ensure_parent(const std::string& path) {
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

void write_tokens(
    std::ostream& output,
    const std::vector<std::vector<int>>& generated_tokens) {
    output << '[';
    for (std::size_t request = 0; request < generated_tokens.size();
         ++request) {
        if (request != 0) output << ',';
        output << '[';
        for (std::size_t token = 0;
             token < generated_tokens[request].size(); ++token) {
            if (token != 0) output << ',';
            output << generated_tokens[request][token];
        }
        output << ']';
    }
    output << ']';
}

void write_json(
    const std::string& path, const cudaDeviceProp& properties,
    int driver_version, int runtime_version,
    std::size_t token_budget, CudaDataType data_type,
    const std::vector<RunResult>& runs) {
    std::vector<double> total_ms;
    std::vector<double> throughput;
    std::vector<double> ttft_p50;
    std::vector<double> ttft_p95;
    std::vector<double> tpot_p50;
    std::vector<double> tpot_p95;
    std::vector<double> latency_p50;
    std::vector<double> latency_p95;
    for (const RunResult& run : runs) {
        total_ms.push_back(run.total_ms);
        throughput.push_back(run.throughput_tokens_per_second);
        ttft_p50.push_back(percentile(run.ttft_ms, 0.5));
        ttft_p95.push_back(percentile(run.ttft_ms, 0.95));
        tpot_p50.push_back(percentile(run.tpot_ms, 0.5));
        tpot_p95.push_back(percentile(run.tpot_ms, 0.95));
        latency_p50.push_back(percentile(run.request_latency_ms, 0.5));
        latency_p95.push_back(percentile(run.request_latency_ms, 0.95));
    }

    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open CUDA JSON");
    file << std::fixed << std::setprecision(6);
    file << "{\n  \"metadata\": {\n"
         << "    \"git_commit\": \"" << MINI_VLLM_GIT_COMMIT << "\",\n"
         << "    \"gpu\": \"" << properties.name << "\",\n"
         << "    \"compute_capability\": \""
         << properties.major << '.' << properties.minor << "\",\n"
         << "    \"compiled_gpu_arch\": \"" << MINI_VLLM_GPU_ARCH
         << "\",\n"
         << "    \"execution_mode\": \"packed_multi_token_prefill\",\n"
         << "    \"precision\": \""
         << (data_type == CudaDataType::FP16 ? "fp16" :
             (data_type == CudaDataType::BF16 ? "bf16" : "fp32"))
         << "\",\n"
         << "    \"scheduler_token_budget\": " << token_budget << ",\n"
         << "    \"driver_version\": " << driver_version << ",\n"
         << "    \"runtime_version\": " << runtime_version << ",\n"
         << "    \"workload\": \"prompt lengths 8/16/24/32, four output tokens each, all arrivals at t=0\",\n"
         << "    \"timing_scope\": \"schedule through synchronous GPU greedy token completion; engine initialization excluded\",\n"
         << "    \"correctness\": \"all generated tokens equal independent CPU full-prefix greedy reference\",\n"
         << "    \"warmup_runs\": 1,\n"
         << "    \"repeats\": " << runs.size() << "\n  },\n"
         << "  \"summary\": {\n"
         << "    \"total_time_median_ms\": "
         << percentile(total_ms, 0.5) << ",\n"
         << "    \"throughput_median_tokens_per_second\": "
         << percentile(throughput, 0.5) << ",\n"
         << "    \"ttft_p50_ms\": " << percentile(ttft_p50, 0.5) << ",\n"
         << "    \"ttft_p95_ms\": " << percentile(ttft_p95, 0.5) << ",\n"
         << "    \"tpot_p50_ms\": " << percentile(tpot_p50, 0.5) << ",\n"
         << "    \"tpot_p95_ms\": " << percentile(tpot_p95, 0.5) << ",\n"
         << "    \"request_latency_p50_ms\": "
         << percentile(latency_p50, 0.5) << ",\n"
         << "    \"request_latency_p95_ms\": "
         << percentile(latency_p95, 0.5) << ",\n"
         << "    \"weight_bytes\": " << runs.front().weight_bytes << ",\n"
         << "    \"kv_cache_bytes\": " << runs.front().kv_cache_bytes
         << ",\n"
         << "    \"activation_bytes\": "
         << runs.front().activation_bytes << "\n  },\n"
         << "  \"runs\": [\n";
    for (std::size_t index = 0; index < runs.size(); ++index) {
        const RunResult& run = runs[index];
        file << "    {\"repeat\": " << index + 1
             << ", \"total_ms\": " << run.total_ms
             << ", \"throughput_tokens_per_second\": "
             << run.throughput_tokens_per_second
             << ", \"metadata_h2d_bytes\": "
             << run.metadata_h2d_bytes
             << ", \"ttft_p50_ms\": "
             << percentile(run.ttft_ms, 0.5)
             << ", \"ttft_p95_ms\": "
             << percentile(run.ttft_ms, 0.95)
             << ", \"tpot_p50_ms\": "
             << percentile(run.tpot_ms, 0.5)
             << ", \"tpot_p95_ms\": "
             << percentile(run.tpot_ms, 0.95)
             << ", \"request_latency_p50_ms\": "
             << percentile(run.request_latency_ms, 0.5)
             << ", \"request_latency_p95_ms\": "
             << percentile(run.request_latency_ms, 0.95)
             << ", \"generated_tokens\": ";
        write_tokens(file, run.generated_tokens);
        file << '}' << (index + 1 == runs.size() ? "\n" : ",\n");
    }
    file << "  ]\n}\n";
}

void write_csv(const std::string& path, const std::vector<RunResult>& runs) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open CUDA CSV");
    file << "repeat,total_ms,throughput_tokens_per_second,metadata_h2d_bytes\n";
    file << std::fixed << std::setprecision(6);
    for (std::size_t index = 0; index < runs.size(); ++index) {
        file << index + 1 << ',' << runs[index].total_ms << ','
             << runs[index].throughput_tokens_per_second << ','
             << runs[index].metadata_h2d_bytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device = 0;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        cuda_check(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        int driver_version = 0;
        int runtime_version = 0;
        cuda_check(
            cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
        cuda_check(
            cudaRuntimeGetVersion(&runtime_version),
            "cudaRuntimeGetVersion");

        GPT2 model{};
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        const std::vector<std::vector<int>> expected_tokens =
            cpu_greedy_reference(model);
        const RunResult warmup = run_once(
            model, options.token_budget, options.data_type);
        if (warmup.generated_tokens != expected_tokens) {
            throw std::runtime_error(
                "CUDA warmup tokens differ from CPU full-prefix reference");
        }
        std::vector<RunResult> runs;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            RunResult run = run_once(
                model, options.token_budget, options.data_type);
            if (run.generated_tokens != warmup.generated_tokens) {
                throw std::runtime_error(
                    "CUDA benchmark generated tokens changed across runs");
            }
            std::cout
                << "repeat=" << repeat + 1
                << " total=" << std::fixed << std::setprecision(3)
                << run.total_ms << " ms throughput="
                << run.throughput_tokens_per_second << " tok/s"
                << " TTFT P50/P95=" << percentile(run.ttft_ms, 0.5)
                << '/' << percentile(run.ttft_ms, 0.95) << " ms"
                << " TPOT P50/P95=" << percentile(run.tpot_ms, 0.5)
                << '/' << percentile(run.tpot_ms, 0.95) << " ms\n";
            runs.push_back(std::move(run));
        }
        ensure_parent(options.json_path);
        ensure_parent(options.csv_path);
        write_json(
            options.json_path, properties, driver_version,
            runtime_version, options.token_budget, options.data_type, runs);
        write_csv(options.csv_path, runs);
        gpt2_free(&model);
        std::cout << "GPU 服务 Benchmark 原始结果已写入 "
                  << options.json_path << " 和 " << options.csv_path
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU 服务 Benchmark 失败：" << error.what() << '\n';
        return 1;
    }
}
