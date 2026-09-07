#ifndef MINI_VLLM_CUDA_PAGED_ATTENTION_CUH
#define MINI_VLLM_CUDA_PAGED_ATTENTION_CUH

#include <cuda_runtime.h>

namespace mini_vllm {
namespace cuda {

constexpr int kPagedAttentionPageSize = 16;

// 数据布局：
// q/new_k/new_v/out: [batch, num_heads, head_size]
// k_cache/v_cache:   [num_pages, num_layers, num_heads, page_size, head_size]
// block_tables:      [batch, max_blocks_per_sequence]
// context_lengths:   [batch]
cudaError_t paged_attention_decode(
    const float* q, const float* new_k, const float* new_v,
    float* k_cache, float* v_cache,
    const int* block_tables, const int* context_lengths,
    float* out, int batch_size, int num_pages, int num_layers,
    int layer_index, int num_heads, int head_size,
    int max_blocks_per_sequence, int max_context_length,
    cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace mini_vllm

#endif
