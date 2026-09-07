#include "paged_attention.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mini_vllm {
namespace cuda {
namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ std::size_t cache_offset(
    int page, int layer, int head, int page_offset, int dimension,
    int num_layers, int num_heads, int head_size) {
    return ((((
        static_cast<std::size_t>(page) * num_layers + layer) *
            num_heads + head) *
            kPagedAttentionPageSize + page_offset) *
            head_size + dimension);
}

__global__ void write_kv_cache_kernel(
    const float* new_k, const float* new_v,
    float* k_cache, float* v_cache,
    const int* context_lengths, const int* slot_mapping,
    int batch_size, int num_layers, int layer_index,
    int num_heads, int head_size) {
    const int request = blockIdx.x;
    const int head = blockIdx.y;
    if (request >= batch_size || head >= num_heads) return;

    if (context_lengths[request] <= 0) return;
    const int physical_slot = slot_mapping[request];
    const int physical_block = physical_slot / kPagedAttentionPageSize;
    const int page_offset = physical_slot % kPagedAttentionPageSize;
    const std::size_t source_base =
        (static_cast<std::size_t>(request) * num_heads + head) *
        head_size;

    for (int dimension = threadIdx.x;
         dimension < head_size; dimension += blockDim.x) {
        const std::size_t destination = cache_offset(
            physical_block, layer_index, head, page_offset, dimension,
            num_layers, num_heads, head_size);
        k_cache[destination] = new_k[source_base + dimension];
        v_cache[destination] = new_v[source_base + dimension];
    }
}

__global__ void paged_attention_kernel(
    const float* q, const float* k_cache, const float* v_cache,
    const int* block_tables, const int* context_lengths,
    float* out, int batch_size, int num_layers, int layer_index,
    int num_heads, int head_size, int max_blocks_per_sequence,
    int max_context_length) {
    const int request = blockIdx.x;
    const int head = blockIdx.y;
    const int thread = threadIdx.x;
    if (request >= batch_size || head >= num_heads) return;

    extern __shared__ float scores[];
    __shared__ float reduction[kThreads];

    const int context_length = context_lengths[request];
    if (context_length <= 0 || context_length > max_context_length) {
        return;
    }
    const std::size_t query_base =
        (static_cast<std::size_t>(request) * num_heads + head) *
        head_size;
    const float scale = rsqrtf(static_cast<float>(head_size));

    float local_max = -FLT_MAX;
    for (int token = thread; token < context_length;
         token += blockDim.x) {
        const int physical_block =
            block_tables[
                request * max_blocks_per_sequence +
                token / kPagedAttentionPageSize];
        const int page_offset = token % kPagedAttentionPageSize;
        float dot = 0.0f;
        for (int dimension = 0; dimension < head_size; ++dimension) {
            const std::size_t offset = cache_offset(
                physical_block, layer_index, head, page_offset,
                dimension, num_layers, num_heads, head_size);
            dot += q[query_base + dimension] * k_cache[offset];
        }
        const float score = dot * scale;
        scores[token] = score;
        local_max = fmaxf(local_max, score);
    }

    reduction[thread] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] =
                fmaxf(reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];
    // Every thread must finish reading the maximum before reduction[] is
    // reused for the softmax sum below.
    __syncthreads();

    float local_sum = 0.0f;
    for (int token = thread; token < context_length;
         token += blockDim.x) {
        const float weight = expf(scores[token] - maximum);
        scores[token] = weight;
        local_sum += weight;
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / reduction[0];
    __syncthreads();

    for (int dimension = thread;
         dimension < head_size; dimension += blockDim.x) {
        float value_sum = 0.0f;
        for (int token = 0; token < context_length; ++token) {
            const int physical_block =
                block_tables[
                    request * max_blocks_per_sequence +
                    token / kPagedAttentionPageSize];
            const int page_offset = token % kPagedAttentionPageSize;
            const std::size_t offset = cache_offset(
                physical_block, layer_index, head, page_offset,
                dimension, num_layers, num_heads, head_size);
            value_sum +=
                scores[token] * inverse_sum * v_cache[offset];
        }
        out[query_base + dimension] = value_sum;
    }
}

} // namespace

cudaError_t paged_attention_decode(
    const float* q, const float* new_k, const float* new_v,
    float* k_cache, float* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    float* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream) {
    if (q == nullptr || new_k == nullptr || new_v == nullptr ||
        k_cache == nullptr || v_cache == nullptr ||
        block_tables == nullptr || context_lengths == nullptr ||
        slot_mapping == nullptr ||
        out == nullptr || batch_size <= 0 || num_pages <= 0 ||
        num_layers <= 0 || layer_index < 0 ||
        layer_index >= num_layers || num_heads <= 0 ||
        head_size <= 0 || max_blocks_per_sequence <= 0 ||
        max_context_length <= 0 ||
        max_blocks_per_sequence * kPagedAttentionPageSize <
            max_context_length) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(batch_size, num_heads);
    write_kv_cache_kernel<<<grid, kThreads, 0, stream>>>(
        new_k, new_v, k_cache, v_cache, context_lengths, slot_mapping,
        batch_size, num_layers, layer_index, num_heads, head_size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    const std::size_t shared_memory =
        static_cast<std::size_t>(max_context_length) * sizeof(float);
    paged_attention_kernel<<<grid, kThreads, shared_memory, stream>>>(
        q, k_cache, v_cache, block_tables, context_lengths, out,
        batch_size, num_layers, layer_index, num_heads, head_size,
        max_blocks_per_sequence, max_context_length);
    return cudaGetLastError();
}

} // namespace cuda
} // namespace mini_vllm
