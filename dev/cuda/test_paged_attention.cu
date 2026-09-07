#include "../../mini_vllm/cuda/paged_attention.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
    const T* get() const { return pointer_; }
    std::size_t count() const { return count_; }

private:
    T* pointer_ = nullptr;
    std::size_t count_;
};

static std::size_t dense_offset(
    int request, int token, int head, int dimension,
    int max_context, int num_heads, int head_size) {
    return (((
        static_cast<std::size_t>(request) * max_context + token) *
        num_heads + head) * head_size + dimension);
}

static std::size_t cache_offset(
    int page, int layer, int head, int page_offset, int dimension,
    int num_layers, int num_heads, int head_size) {
    return ((((
        static_cast<std::size_t>(page) * num_layers + layer) *
        num_heads + head) * kPagedAttentionPageSize + page_offset) *
        head_size + dimension);
}

static std::vector<float> dense_reference(
    const std::vector<float>& query,
    const std::vector<float>& dense_keys,
    const std::vector<float>& dense_values,
    const std::vector<int>& context_lengths,
    int batch_size, int max_context, int num_heads, int head_size) {
    std::vector<float> output(
        static_cast<std::size_t>(batch_size) * num_heads * head_size);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));
    for (int request = 0; request < batch_size; ++request) {
        for (int head = 0; head < num_heads; ++head) {
            std::vector<double> scores(context_lengths[request]);
            double maximum = -std::numeric_limits<double>::infinity();
            const std::size_t query_base =
                (static_cast<std::size_t>(request) * num_heads + head) *
                head_size;
            for (int token = 0; token < context_lengths[request]; ++token) {
                double dot = 0.0;
                for (int dimension = 0; dimension < head_size; ++dimension) {
                    dot += static_cast<double>(
                               query[query_base + dimension]) *
                           dense_keys[dense_offset(
                               request, token, head, dimension, max_context,
                               num_heads, head_size)];
                }
                scores[token] = dot * scale;
                maximum = std::max(maximum, scores[token]);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int dimension = 0; dimension < head_size; ++dimension) {
                double sum = 0.0;
                for (int token = 0;
                     token < context_lengths[request]; ++token) {
                    sum += scores[token] / denominator *
                           dense_values[dense_offset(
                               request, token, head, dimension, max_context,
                               num_heads, head_size)];
                }
                output[query_base + dimension] =
                    static_cast<float>(sum);
            }
        }
    }
    return output;
}

static void copy_to_device(
    DeviceBuffer<float>& destination,
    const std::vector<float>& source) {
    CUDA_CHECK(cudaMemcpy(
        destination.get(), source.data(), source.size() * sizeof(float),
        cudaMemcpyHostToDevice));
}

static void copy_to_device(
    DeviceBuffer<int>& destination,
    const std::vector<int>& source) {
    CUDA_CHECK(cudaMemcpy(
        destination.get(), source.data(), source.size() * sizeof(int),
        cudaMemcpyHostToDevice));
}

int main() {
    try {
        constexpr int batch_size = 3;
        constexpr int num_layers = 2;
        constexpr int layer_index = 1;
        constexpr int num_heads = 4;
        constexpr int head_size = 64;
        constexpr int max_context = 64;
        constexpr int max_blocks =
            max_context / kPagedAttentionPageSize;
        constexpr int num_pages = batch_size * max_blocks;

        const std::vector<int> block_tables = {
            11, 2, 7, 0,
            5, 9, 1, 10,
            3, 8, 4, 6,
        };
        const std::array<std::array<int, batch_size>, 3> cases = {{
            {{1, 15, 16}},
            {{17, 31, 32}},
            {{33, 64, 7}},
        }};

        std::mt19937 generator(20260907);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        const std::size_t qkv_elements =
            static_cast<std::size_t>(batch_size) * num_heads * head_size;
        const std::size_t dense_elements =
            static_cast<std::size_t>(batch_size) * max_context *
            num_heads * head_size;
        const std::size_t cache_elements =
            static_cast<std::size_t>(num_pages) * num_layers *
            num_heads * kPagedAttentionPageSize * head_size;

        std::vector<float> query(qkv_elements);
        std::vector<float> dense_keys(dense_elements);
        std::vector<float> dense_values(dense_elements);
        for (float& value : query) value = distribution(generator);
        for (float& value : dense_keys) value = distribution(generator);
        for (float& value : dense_values) value = distribution(generator);

        DeviceBuffer<float> device_query(qkv_elements);
        DeviceBuffer<float> device_new_key(qkv_elements);
        DeviceBuffer<float> device_new_value(qkv_elements);
        DeviceBuffer<float> device_key_cache(cache_elements);
        DeviceBuffer<float> device_value_cache(cache_elements);
        DeviceBuffer<float> device_output(qkv_elements);
        DeviceBuffer<int> device_block_tables(block_tables.size());
        DeviceBuffer<int> device_context_lengths(batch_size);
        copy_to_device(device_query, query);
        copy_to_device(device_block_tables, block_tables);

        double global_max_abs_error = 0.0;
        double global_max_rel_error = 0.0;
        double global_max_kv_write_error = 0.0;

        for (const auto& lengths : cases) {
            const std::vector<int> context_lengths(
                lengths.begin(), lengths.end());
            std::vector<float> key_cache(cache_elements, 0.0f);
            std::vector<float> value_cache(cache_elements, 0.0f);
            std::vector<float> new_key(qkv_elements);
            std::vector<float> new_value(qkv_elements);

            for (int request = 0; request < batch_size; ++request) {
                for (int token = 0; token < context_lengths[request];
                     ++token) {
                    const int physical_page =
                        block_tables[
                            request * max_blocks +
                            token / kPagedAttentionPageSize];
                    const int page_offset =
                        token % kPagedAttentionPageSize;
                    for (int head = 0; head < num_heads; ++head) {
                        for (int dimension = 0;
                             dimension < head_size; ++dimension) {
                            const std::size_t dense = dense_offset(
                                request, token, head, dimension,
                                max_context, num_heads, head_size);
                            const std::size_t cache = cache_offset(
                                physical_page, layer_index, head,
                                page_offset, dimension, num_layers,
                                num_heads, head_size);
                            key_cache[cache] = dense_keys[dense];
                            value_cache[cache] = dense_values[dense];
                            if (token == context_lengths[request] - 1) {
                                const std::size_t current =
                                    (static_cast<std::size_t>(request) *
                                         num_heads +
                                     head) *
                                        head_size +
                                    dimension;
                                new_key[current] = dense_keys[dense];
                                new_value[current] = dense_values[dense];
                                key_cache[cache] = 0.0f;
                                value_cache[cache] = 0.0f;
                            }
                        }
                    }
                }
            }

            const std::vector<float> expected = dense_reference(
                query, dense_keys, dense_values, context_lengths,
                batch_size, max_context, num_heads, head_size);
            copy_to_device(device_new_key, new_key);
            copy_to_device(device_new_value, new_value);
            copy_to_device(device_key_cache, key_cache);
            copy_to_device(device_value_cache, value_cache);
            copy_to_device(device_context_lengths, context_lengths);

            CUDA_CHECK(paged_attention_decode(
                device_query.get(), device_new_key.get(),
                device_new_value.get(), device_key_cache.get(),
                device_value_cache.get(), device_block_tables.get(),
                device_context_lengths.get(), device_output.get(),
                batch_size, num_pages, num_layers, layer_index,
                num_heads, head_size, max_blocks, max_context));
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> actual(qkv_elements);
            CUDA_CHECK(cudaMemcpy(
                actual.data(), device_output.get(),
                actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                key_cache.data(), device_key_cache.get(),
                key_cache.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                value_cache.data(), device_value_cache.get(),
                value_cache.size() * sizeof(float), cudaMemcpyDeviceToHost));

            for (std::size_t index = 0; index < actual.size(); ++index) {
                const double absolute_error = std::abs(
                    static_cast<double>(actual[index]) - expected[index]);
                const double relative_error =
                    absolute_error /
                    std::max(1e-6, std::abs(
                        static_cast<double>(expected[index])));
                global_max_abs_error =
                    std::max(global_max_abs_error, absolute_error);
                global_max_rel_error =
                    std::max(global_max_rel_error, relative_error);
            }
            for (int request = 0; request < batch_size; ++request) {
                const int token = context_lengths[request] - 1;
                const int physical_page =
                    block_tables[
                        request * max_blocks +
                        token / kPagedAttentionPageSize];
                const int page_offset =
                    token % kPagedAttentionPageSize;
                for (int head = 0; head < num_heads; ++head) {
                    for (int dimension = 0;
                         dimension < head_size; ++dimension) {
                        const std::size_t dense = dense_offset(
                            request, token, head, dimension, max_context,
                            num_heads, head_size);
                        const std::size_t cache = cache_offset(
                            physical_page, layer_index, head, page_offset,
                            dimension, num_layers, num_heads, head_size);
                        global_max_kv_write_error = std::max(
                            global_max_kv_write_error,
                            std::abs(static_cast<double>(key_cache[cache]) -
                                     dense_keys[dense]));
                        global_max_kv_write_error = std::max(
                            global_max_kv_write_error,
                            std::abs(static_cast<double>(value_cache[cache]) -
                                     dense_values[dense]));
                    }
                }
            }
        }

        std::cout
            << "CUDA PagedAttention correctness passed: "
            << "lengths=1,7,15,16,17,31,32,33,64 "
            << "max_abs_error=" << global_max_abs_error
            << " max_rel_error=" << global_max_rel_error
            << " max_kv_write_error=" << global_max_kv_write_error
            << '\n';
        return global_max_abs_error < 2e-4 &&
                       global_max_kv_write_error == 0.0
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::cerr << "CUDA PagedAttention test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
