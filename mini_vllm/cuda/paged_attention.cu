#include "paged_attention.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

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

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void write_kv_cache_kernel(
    const T* new_k, const T* new_v,
    T* k_cache, T* v_cache,
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

    if constexpr (std::is_same<T, __half>::value) {
        if (head_size % 2 == 0) {
            const __half2* source_k =
                reinterpret_cast<const __half2*>(new_k) + source_base / 2;
            const __half2* source_v =
                reinterpret_cast<const __half2*>(new_v) + source_base / 2;
            __half2* destination_k = reinterpret_cast<__half2*>(k_cache);
            __half2* destination_v = reinterpret_cast<__half2*>(v_cache);
            for (int pair = threadIdx.x; pair < head_size / 2;
                 pair += blockDim.x) {
                const std::size_t destination = cache_offset(
                    physical_block, layer_index, head, page_offset,
                    pair * 2, num_layers, num_heads, head_size);
                destination_k[destination / 2] = source_k[pair];
                destination_v[destination / 2] = source_v[pair];
            }
            return;
        }
    }

    for (int dimension = threadIdx.x;
         dimension < head_size; dimension += blockDim.x) {
        const std::size_t destination = cache_offset(
            physical_block, layer_index, head, page_offset, dimension,
            num_layers, num_heads, head_size);
        k_cache[destination] = new_k[source_base + dimension];
        v_cache[destination] = new_v[source_base + dimension];
    }
}

template <typename T>
__global__ void paged_attention_kernel(
    const T* q, const T* k_cache, const T* v_cache,
    const int* block_tables, const int* context_lengths,
    T* out, int batch_size, int num_layers, int layer_index,
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
        if constexpr (std::is_same<T, __half>::value) {
            if (head_size % 2 == 0) {
                const __half2* query2 =
                    reinterpret_cast<const __half2*>(q) + query_base / 2;
                const __half2* cache2 =
                    reinterpret_cast<const __half2*>(k_cache);
                for (int pair = 0; pair < head_size / 2; ++pair) {
                    const std::size_t offset = cache_offset(
                        physical_block, layer_index, head, page_offset,
                        pair * 2, num_layers, num_heads, head_size);
                    const float2 q_pair = __half22float2(query2[pair]);
                    const float2 k_pair = __half22float2(cache2[offset / 2]);
                    dot += q_pair.x * k_pair.x + q_pair.y * k_pair.y;
                }
            } else {
                for (int dimension = 0; dimension < head_size; ++dimension) {
                    const std::size_t offset = cache_offset(
                        physical_block, layer_index, head, page_offset,
                        dimension, num_layers, num_heads, head_size);
                    dot += to_float(q[query_base + dimension]) *
                        to_float(k_cache[offset]);
                }
            }
        } else {
            for (int dimension = 0; dimension < head_size; ++dimension) {
                const std::size_t offset = cache_offset(
                    physical_block, layer_index, head, page_offset,
                    dimension, num_layers, num_heads, head_size);
                dot += to_float(q[query_base + dimension]) *
                    to_float(k_cache[offset]);
            }
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

    if constexpr (std::is_same<T, __half>::value) {
        if (head_size % 2 == 0) {
            const __half2* cache2 =
                reinterpret_cast<const __half2*>(v_cache);
            __half2* output2 = reinterpret_cast<__half2*>(out) +
                query_base / 2;
            for (int pair = thread; pair < head_size / 2;
                 pair += blockDim.x) {
                float2 value_sum = make_float2(0.0f, 0.0f);
                for (int token = 0; token < context_length; ++token) {
                    const int physical_block = block_tables[
                        request * max_blocks_per_sequence +
                        token / kPagedAttentionPageSize];
                    const int page_offset = token % kPagedAttentionPageSize;
                    const std::size_t offset = cache_offset(
                        physical_block, layer_index, head, page_offset,
                        pair * 2, num_layers, num_heads, head_size);
                    const float2 cached =
                        __half22float2(cache2[offset / 2]);
                    const float weight = scores[token] * inverse_sum;
                    value_sum.x += weight * cached.x;
                    value_sum.y += weight * cached.y;
                }
                output2[pair] = __floats2half2_rn(
                    value_sum.x, value_sum.y);
            }
            return;
        }
    }

    for (int dimension = thread; dimension < head_size;
         dimension += blockDim.x) {
        float value_sum = 0.0f;
        for (int token = 0; token < context_length; ++token) {
            const int physical_block = block_tables[
                request * max_blocks_per_sequence +
                token / kPagedAttentionPageSize];
            const int page_offset = token % kPagedAttentionPageSize;
            const std::size_t offset = cache_offset(
                physical_block, layer_index, head, page_offset,
                dimension, num_layers, num_heads, head_size);
            value_sum += scores[token] * inverse_sum *
                to_float(v_cache[offset]);
        }
        out[query_base + dimension] = from_float<T>(value_sum);
    }
}

template <typename T>
cudaError_t paged_attention_decode_impl(
    const T* q, const T* new_k, const T* new_v,
    T* k_cache, T* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    T* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream) {
    if (q == nullptr || new_k == nullptr || new_v == nullptr ||
        k_cache == nullptr || v_cache == nullptr ||
        block_tables == nullptr || context_lengths == nullptr ||
        slot_mapping == nullptr || out == nullptr || batch_size <= 0 ||
        num_pages <= 0 || num_layers <= 0 || layer_index < 0 ||
        layer_index >= num_layers || num_heads <= 0 || head_size <= 0 ||
        max_blocks_per_sequence <= 0 || max_context_length <= 0 ||
        max_blocks_per_sequence * kPagedAttentionPageSize <
            max_context_length) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(batch_size, num_heads);
    write_kv_cache_kernel<T><<<grid, kThreads, 0, stream>>>(
        new_k, new_v, k_cache, v_cache, context_lengths, slot_mapping,
        batch_size, num_layers, layer_index, num_heads, head_size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    const std::size_t shared_memory =
        static_cast<std::size_t>(max_context_length) * sizeof(float);
    paged_attention_kernel<T><<<grid, kThreads, shared_memory, stream>>>(
        q, k_cache, v_cache, block_tables, context_lengths, out,
        batch_size, num_layers, layer_index, num_heads, head_size,
        max_blocks_per_sequence, max_context_length);
    return cudaGetLastError();
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
    return paged_attention_decode_impl(
        q, new_k, new_v, k_cache, v_cache, block_tables,
        context_lengths, slot_mapping, out, batch_size, num_pages,
        num_layers, layer_index, num_heads, head_size,
        max_blocks_per_sequence, max_context_length, stream);
}

cudaError_t paged_attention_decode(
    const __nv_bfloat16* q, const __nv_bfloat16* new_k,
    const __nv_bfloat16* new_v, __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    __nv_bfloat16* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream) {
    return paged_attention_decode_impl(
        q, new_k, new_v, k_cache, v_cache, block_tables,
        context_lengths, slot_mapping, out, batch_size, num_pages,
        num_layers, layer_index, num_heads, head_size,
        max_blocks_per_sequence, max_context_length, stream);
}

cudaError_t paged_attention_decode(
    const __half* q, const __half* new_k, const __half* new_v,
    __half* k_cache, __half* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    __half* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream) {
    return paged_attention_decode_impl(
        q, new_k, new_v, k_cache, v_cache, block_tables,
        context_lengths, slot_mapping, out, batch_size, num_pages,
        num_layers, layer_index, num_heads, head_size,
        max_blocks_per_sequence, max_context_length, stream);
}

} // namespace cuda
} // namespace mini_vllm
