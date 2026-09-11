#ifndef MINI_VLLM_CUDA_GPT2_CUDA_MODEL_RUNNER_CUH
#define MINI_VLLM_CUDA_GPT2_CUDA_MODEL_RUNNER_CUH

#include "../model_input.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace mini_vllm {
namespace cuda {

enum class CudaDataType {
    FP32,
    FP16,
    BF16,
};

struct GPT2CudaConfig {
    int max_seq_len = 0;
    int vocab_size = 0;
    int padded_vocab_size = 0;
    int num_layers = 0;
    int num_heads = 0;
    int channels = 0;
    CudaDataType data_type = CudaDataType::FP32;
    bool enable_fused_residual_layernorm = false;
    bool enable_cuda_graph = false;
    bool enable_sample_row_pruning = true;
    int device_id = -1; // -1：构造时的当前设备；之后 Runner 固定归属该设备。
};

struct KVTransferStats {
    std::size_t payload_bytes = 0; // K+V，含最后一页未使用的槽位
    std::size_t num_pages = 0;
    double d2h_ms = 0.0;
    double h2d_ms = 0.0;
    double total_ms = 0.0; // 含 pinned staging 分配/释放及设备切换
};

class GPT2CudaModelRunner {
public:
    GPT2CudaModelRunner(
        GPT2CudaConfig config, const float* host_parameters,
        std::size_t num_parameters, BlockManager& block_manager,
        std::size_t max_num_sequences,
        std::size_t max_num_batched_tokens,
        std::size_t max_context_length);
    ~GPT2CudaModelRunner();

    GPT2CudaModelRunner(const GPT2CudaModelRunner&) = delete;
    GPT2CudaModelRunner& operator=(const GPT2CudaModelRunner&) = delete;

    std::vector<int> run(const SchedulerOutput& output);

    const std::vector<ModelInput>& last_model_inputs() const;
    std::vector<float> last_logits_for_testing() const;
    // 第 i 行 logits 对应 last_model_inputs()[0] 中的哪个 Packed Token。
    const std::vector<int>& last_logit_token_indices() const;
    std::size_t last_host_to_device_bytes() const;
    std::size_t weight_bytes() const;
    std::size_t kv_cache_bytes() const;
    std::size_t activation_bytes() const;
    CudaDataType data_type() const;
    std::size_t num_cuda_graphs() const;
    int device_id() const { return device_id_; }

    // 调用方须保证两端权重相同、独占目标页，且两个 Runner 此刻均无并发 run。
    // 只复制已计算 Token 覆盖的物理页；不更改 Sequence 状态或释放源页。
    KVTransferStats copy_kv_to(
        GPT2CudaModelRunner& destination, const Sequence& source,
        const Sequence& target, std::size_t computed_tokens);

private:
    class Impl;
    int device_id_ = -1;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuda
} // namespace mini_vllm

#endif
