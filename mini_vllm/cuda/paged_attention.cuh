#ifndef MINI_VLLM_CUDA_PAGED_ATTENTION_CUH
#define MINI_VLLM_CUDA_PAGED_ATTENTION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace mini_vllm {
namespace cuda {

constexpr int kPagedAttentionPageSize = 16;

// 数据布局：
// q/new_k/new_v/out: [batch, num_heads, head_size]
// k_cache/v_cache:   [num_pages, num_layers, num_heads, page_size, head_size]
// block_tables:      [batch, max_blocks_per_sequence]
// context_lengths:   [batch]
// slot_mapping:      [batch]，值为 physical_page * page_size + offset
cudaError_t paged_attention_decode(
    const float* q, const float* new_k, const float* new_v,
    float* k_cache, float* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    float* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream = nullptr);

// BF16 与 FP16 使用相同的 FP32 Attention 归约策略。
cudaError_t paged_attention_decode(
    const __nv_bfloat16* q, const __nv_bfloat16* new_k,
    const __nv_bfloat16* new_v, __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    __nv_bfloat16* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream = nullptr);

// FP16 存储路径仍使用 FP32 计算 QK 点积、Softmax 和 V 加权归约。
cudaError_t paged_attention_decode(
    const __half* q, const __half* new_k, const __half* new_v,
    __half* k_cache, __half* v_cache,
    const int* block_tables, const int* context_lengths,
    const int* slot_mapping,
    __half* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace mini_vllm

#endif
