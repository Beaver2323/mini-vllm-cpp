#include "../mini_vllm/cuda/paged_attention.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MINI_VLLM_GIT_COMMIT
#define MINI_VLLM_GIT_COMMIT "unknown"
#endif

#ifndef MINI_VLLM_GPU_ARCH
#define MINI_VLLM_GPU_ARCH "unknown"
#endif

using mini_vllm::cuda::kPagedAttentionPageSize;
using mini_vllm::cuda::paged_attention_decode;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t error = (call);                                    \
        if (error != cudaSuccess) {                                          \
            throw std::runtime_error(                                        \
                std::string(#call) + ": " + cudaGetErrorString(error));     \
        }                                                                    \
    } while (0)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&pointer_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return pointer_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* pointer_ = nullptr;
    std::size_t count_;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

struct Options {
    int warmup_iterations = 20;
    int iterations_per_repeat = 50;
    int repeats = 20;
    std::string json_path =
        "benchmark/results/cuda_paged_attention_rtx3090.json";
    std::string csv_path =
        "benchmark/results/cuda_paged_attention_rtx3090.csv";
};

struct Result {
    int batch_size;
    int context_length;
    double latency_p50_us;
    double latency_p95_us;
    double useful_bandwidth_gbps;
};

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

static Result benchmark_case(
    int batch_size, int context_length, const Options& options) {
    constexpr int num_layers = 1;
    constexpr int layer_index = 0;
    constexpr int num_heads = 12;
    constexpr int head_size = 64;
    const int max_blocks =
        (context_length + kPagedAttentionPageSize - 1) /
        kPagedAttentionPageSize;
    const int num_pages = batch_size * max_blocks;
    const std::size_t qkv_elements =
        static_cast<std::size_t>(batch_size) * num_heads * head_size;
    const std::size_t cache_elements =
        static_cast<std::size_t>(num_pages) * num_layers * num_heads *
        kPagedAttentionPageSize * head_size;

    DeviceBuffer<float> query(qkv_elements);
    DeviceBuffer<float> new_key(qkv_elements);
    DeviceBuffer<float> new_value(qkv_elements);
    DeviceBuffer<float> key_cache(cache_elements);
    DeviceBuffer<float> value_cache(cache_elements);
    DeviceBuffer<float> output(qkv_elements);
    DeviceBuffer<int> block_tables(
        static_cast<std::size_t>(batch_size) * max_blocks);
    DeviceBuffer<int> context_lengths(batch_size);
    DeviceBuffer<int> slot_mapping(batch_size);

    CUDA_CHECK(cudaMemset(query.get(), 0x3f, query.bytes()));
    CUDA_CHECK(cudaMemset(new_key.get(), 0x3e, new_key.bytes()));
    CUDA_CHECK(cudaMemset(new_value.get(), 0x3d, new_value.bytes()));
    CUDA_CHECK(cudaMemset(key_cache.get(), 0x3e, key_cache.bytes()));
    CUDA_CHECK(cudaMemset(value_cache.get(), 0x3d, value_cache.bytes()));

    std::vector<int> host_block_tables(
        static_cast<std::size_t>(batch_size) * max_blocks);
    for (int request = 0; request < batch_size; ++request) {
        for (int block = 0; block < max_blocks; ++block) {
            host_block_tables[
                request * max_blocks + block] =
                request * max_blocks + block;
        }
    }
    std::vector<int> host_context_lengths(
        batch_size, context_length);
    std::vector<int> host_slot_mapping(batch_size);
    for (int request = 0; request < batch_size; ++request) {
        host_slot_mapping[request] =
            (request * max_blocks + max_blocks - 1) *
                kPagedAttentionPageSize +
            (context_length - 1) % kPagedAttentionPageSize;
    }
    CUDA_CHECK(cudaMemcpy(
        block_tables.get(), host_block_tables.data(),
        block_tables.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        context_lengths.get(), host_context_lengths.data(),
        context_lengths.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        slot_mapping.get(), host_slot_mapping.data(),
        slot_mapping.bytes(), cudaMemcpyHostToDevice));

    for (int i = 0; i < options.warmup_iterations; ++i) {
        CUDA_CHECK(paged_attention_decode(
            query.get(), new_key.get(), new_value.get(),
            key_cache.get(), value_cache.get(), block_tables.get(),
            context_lengths.get(), slot_mapping.get(), output.get(),
            batch_size, num_pages,
            num_layers, layer_index, num_heads, head_size, max_blocks,
            context_length));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> repeat_latency_us;
    CudaEvent start;
    CudaEvent end;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        CUDA_CHECK(cudaEventRecord(start.get()));
        for (int iteration = 0;
             iteration < options.iterations_per_repeat; ++iteration) {
            CUDA_CHECK(paged_attention_decode(
                query.get(), new_key.get(), new_value.get(),
                key_cache.get(), value_cache.get(), block_tables.get(),
                context_lengths.get(), slot_mapping.get(), output.get(),
                batch_size, num_pages,
                num_layers, layer_index, num_heads, head_size, max_blocks,
                context_length));
        }
        CUDA_CHECK(cudaEventRecord(end.get()));
        CUDA_CHECK(cudaEventSynchronize(end.get()));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, start.get(), end.get()));
        repeat_latency_us.push_back(
            elapsed_ms * 1000.0 /
            options.iterations_per_repeat);
    }

    const double latency_p50_us =
        percentile(repeat_latency_us, 0.50);
    const double latency_p95_us =
        percentile(repeat_latency_us, 0.95);
    // “有效字节”只计算算法必须读取/写入一次的数据：
    // Q、全部 K/V、输出和当前新 K/V，不计算重复加载及缓存命中。
    const double useful_bytes =
        static_cast<double>(sizeof(float)) * batch_size * num_heads *
        head_size * (2.0 * context_length + 4.0);
    const double useful_bandwidth_gbps =
        useful_bytes / (latency_p50_us * 1000.0);
    return {
        batch_size, context_length, latency_p50_us, latency_p95_us,
        useful_bandwidth_gbps};
}

static std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (char character : value) {
        if (character == '\\') output << "\\\\";
        else if (character == '"') output << "\\\"";
        else if (character == '\n') output << "\\n";
        else output << character;
    }
    return output.str();
}

static void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

static void write_csv(
    const std::string& path, const std::vector<Result>& results) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open CUDA CSV");
    file << "batch_size,context_length,latency_p50_us,latency_p95_us,"
            "useful_bandwidth_gbps\n";
    file << std::fixed << std::setprecision(6);
    for (const Result& result : results) {
        file << result.batch_size << ',' << result.context_length << ','
             << result.latency_p50_us << ',' << result.latency_p95_us
             << ',' << result.useful_bandwidth_gbps << '\n';
    }
}

static void write_json(
    const std::string& path, const Options& options,
    const cudaDeviceProp& properties, int driver_version,
    int runtime_version, const std::vector<Result>& results) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open CUDA JSON");
    file << std::fixed << std::setprecision(6);
    file << "{\n"
         << "  \"metadata\": {\n"
         << "    \"git_commit\": \"" << MINI_VLLM_GIT_COMMIT << "\",\n"
         << "    \"gpu\": \"" << json_escape(properties.name) << "\",\n"
         << "    \"compute_capability\": \""
         << properties.major << '.' << properties.minor << "\",\n"
         << "    \"compiled_gpu_arch\": \""
         << MINI_VLLM_GPU_ARCH << "\",\n"
         << "    \"driver_version\": " << driver_version << ",\n"
         << "    \"runtime_version\": " << runtime_version << ",\n"
         << "    \"num_heads\": 12,\n"
         << "    \"head_size\": 64,\n"
         << "    \"page_size\": " << kPagedAttentionPageSize << ",\n"
         << "    \"warmup_iterations\": "
         << options.warmup_iterations << ",\n"
         << "    \"iterations_per_repeat\": "
         << options.iterations_per_repeat << ",\n"
         << "    \"repeats\": " << options.repeats << ",\n"
         << "    \"latency_scope\": "
            "\"two GPU kernels: KV write plus paged attention\"\n"
         << "  },\n  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const Result& result = results[i];
        file << "    {\"batch_size\": " << result.batch_size
             << ", \"context_length\": " << result.context_length
             << ", \"latency_p50_us\": " << result.latency_p50_us
             << ", \"latency_p95_us\": " << result.latency_p95_us
             << ", \"useful_bandwidth_gbps\": "
             << result.useful_bandwidth_gbps << "}"
             << (i + 1 == results.size() ? "\n" : ",\n");
    }
    file << "  ]\n}\n";
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "missing value for " + argument);
            }
            return argv[++i];
        };
        if (argument == "--warmup") {
            options.warmup_iterations = std::stoi(value());
        } else if (argument == "--iterations") {
            options.iterations_per_repeat = std::stoi(value());
        } else if (argument == "--repeats") {
            options.repeats = std::stoi(value());
        } else if (argument == "--json") {
            options.json_path = value();
        } else if (argument == "--csv") {
            options.csv_path = value();
        } else {
            throw std::invalid_argument(
                "unknown CUDA benchmark option: " + argument);
        }
    }
    if (options.warmup_iterations < 0 ||
        options.iterations_per_repeat <= 0 || options.repeats <= 0) {
        throw std::invalid_argument("invalid CUDA benchmark count");
    }
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        int driver_version = 0;
        int runtime_version = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        const std::array<int, 3> batch_sizes = {1, 8, 32};
        const std::array<int, 4> context_lengths = {16, 64, 256, 512};
        std::vector<Result> results;
        for (int batch_size : batch_sizes) {
            for (int context_length : context_lengths) {
                const Result result =
                    benchmark_case(batch_size, context_length, options);
                results.push_back(result);
                std::cout << "B=" << batch_size
                          << " context=" << context_length
                          << " P50=" << std::fixed << std::setprecision(3)
                          << result.latency_p50_us << " us"
                          << " P95=" << result.latency_p95_us << " us"
                          << " useful_bandwidth="
                          << result.useful_bandwidth_gbps << " GB/s\n";
            }
        }

        ensure_parent_directory(options.json_path);
        ensure_parent_directory(options.csv_path);
        write_json(
            options.json_path, options, properties, driver_version,
            runtime_version, results);
        write_csv(options.csv_path, results);
        std::cout << "CUDA 原始结果已写入 " << options.json_path
                  << " 和 " << options.csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Benchmark 失败：" << error.what() << '\n';
        return 1;
    }
}
