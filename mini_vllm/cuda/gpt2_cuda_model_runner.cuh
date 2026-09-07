#ifndef MINI_VLLM_CUDA_GPT2_CUDA_MODEL_RUNNER_CUH
#define MINI_VLLM_CUDA_GPT2_CUDA_MODEL_RUNNER_CUH

#include "../model_input.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace mini_vllm {
namespace cuda {

struct GPT2CudaConfig {
    int max_seq_len = 0;
    int vocab_size = 0;
    int padded_vocab_size = 0;
    int num_layers = 0;
    int num_heads = 0;
    int channels = 0;
};

class GPT2CudaModelRunner {
public:
    GPT2CudaModelRunner(
        GPT2CudaConfig config, const float* host_parameters,
        std::size_t num_parameters, BlockManager& block_manager,
        std::size_t max_num_sequences,
        std::size_t max_context_length);
    ~GPT2CudaModelRunner();

    GPT2CudaModelRunner(const GPT2CudaModelRunner&) = delete;
    GPT2CudaModelRunner& operator=(const GPT2CudaModelRunner&) = delete;

    std::vector<int> run(const SchedulerOutput& output);

    const std::vector<ModelInput>& last_model_inputs() const;
    std::vector<float> last_logits_for_testing() const;
    std::size_t last_host_to_device_bytes() const;
    std::size_t weight_bytes() const;
    std::size_t kv_cache_bytes() const;
    std::size_t activation_bytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuda
} // namespace mini_vllm

#endif
