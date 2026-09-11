#include "gpt2_cuda_model_runner.cuh"
#include "paged_attention.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_vllm {
namespace cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kParameterTensorCount = 16;

std::size_t storage_element_size(CudaDataType data_type) {
    switch (data_type) {
    case CudaDataType::FP32:
        return sizeof(float);
    case CudaDataType::FP16:
        return sizeof(__half);
    case CudaDataType::BF16:
        return sizeof(__nv_bfloat16);
    }
    throw std::invalid_argument("unsupported CUDA data type");
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

// CUDA 当前设备是线程局部状态；每次入口设置并恢复，避免跨卡析构/执行。
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        check_cuda(cudaGetDevice(&previous_), "get current device");
        if (device != previous_) check_cuda(cudaSetDevice(device), "set runner device");
    }
    ~DeviceGuard() { cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count == 0) throw std::invalid_argument("zero-sized CUDA buffer");
        check_cuda(
            cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }
    const T* get() const { return pointer_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

class DeviceTensorBuffer {
public:
    DeviceTensorBuffer(std::size_t count, std::size_t element_size)
        : count_(count), element_size_(element_size) {
        if (count == 0 || (element_size != sizeof(float) &&
                          element_size != sizeof(__half))) {
            throw std::invalid_argument("invalid CUDA tensor buffer");
        }
        check_cuda(cudaMalloc(&pointer_, bytes()), "cudaMalloc tensor");
    }

    ~DeviceTensorBuffer() {
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

    DeviceTensorBuffer(const DeviceTensorBuffer&) = delete;
    DeviceTensorBuffer& operator=(const DeviceTensorBuffer&) = delete;

    template <typename T>
    T* get() {
        if (sizeof(T) != element_size_) {
            throw std::logic_error("CUDA tensor type does not match storage");
        }
        return static_cast<T*>(pointer_);
    }

    template <typename T>
    const T* get() const {
        if (sizeof(T) != element_size_) {
            throw std::logic_error("CUDA tensor type does not match storage");
        }
        return static_cast<const T*>(pointer_);
    }

    void* data() { return pointer_; }
    std::size_t bytes() const { return count_ * element_size_; }

private:
    void* pointer_ = nullptr;
    std::size_t count_ = 0;
    std::size_t element_size_ = 0;
};

class CudaStream {
public:
    CudaStream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }
    ~CudaStream() {
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle(cudaStream_t stream, CudaDataType data_type) {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
        check_cublas(
            cublasSetMathMode(
                handle_, data_type != CudaDataType::FP32
                    ? CUBLAS_DEFAULT_MATH
                    : CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode");
    }
    ~CublasHandle() {
        if (handle_ != nullptr) cublasDestroy(handle_);
    }
    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

template <typename T>
struct ParameterViews {
    const T* wte = nullptr;
    const T* wpe = nullptr;
    const T* ln1w = nullptr;
    const T* ln1b = nullptr;
    const T* qkvw = nullptr;
    const T* qkvb = nullptr;
    const T* attprojw = nullptr;
    const T* attprojb = nullptr;
    const T* ln2w = nullptr;
    const T* ln2b = nullptr;
    const T* fcw = nullptr;
    const T* fcb = nullptr;
    const T* fcprojw = nullptr;
    const T* fcprojb = nullptr;
    const T* lnfw = nullptr;
    const T* lnfb = nullptr;
};

std::vector<std::size_t> parameter_sizes(const GPT2CudaConfig& config) {
    const std::size_t vp = config.padded_vocab_size;
    const std::size_t c = config.channels;
    const std::size_t max_t = config.max_seq_len;
    const std::size_t l = config.num_layers;
    return {
        vp * c, max_t * c, l * c, l * c,
        l * 3 * c * c, l * 3 * c, l * c * c, l * c,
        l * c, l * c, l * 4 * c * c, l * 4 * c,
        l * c * 4 * c, l * c, c, c,
    };
}

std::size_t parameter_count(const GPT2CudaConfig& config) {
    std::size_t result = 0;
    for (std::size_t size : parameter_sizes(config)) result += size;
    return result;
}

GPT2CudaConfig validate_runner_arguments(
    GPT2CudaConfig config, const float* host_parameters,
    std::size_t supplied_parameters, const BlockManager& block_manager,
    std::size_t max_num_sequences, std::size_t max_num_batched_tokens,
    std::size_t max_context_length) {
    if (config.max_seq_len <= 0 || config.vocab_size <= 0 ||
        config.padded_vocab_size < config.vocab_size ||
        config.num_layers <= 0 || config.num_heads <= 0 ||
        config.channels <= 0 ||
        config.channels % config.num_heads != 0) {
        throw std::invalid_argument("invalid CUDA GPT-2 config");
    }
    storage_element_size(config.data_type);
    if (host_parameters == nullptr ||
        supplied_parameters != parameter_count(config)) {
        throw std::invalid_argument(
            "CUDA runner received an invalid GPT-2 parameter buffer");
    }
    if (max_num_sequences == 0 || max_num_batched_tokens == 0 ||
        max_context_length == 0 ||
        max_context_length >
            static_cast<std::size_t>(config.max_seq_len)) {
        throw std::invalid_argument("invalid CUDA runner capacity");
    }
    if (block_manager.block_size() != kPagedAttentionPageSize) {
        throw std::invalid_argument(
            "BlockManager block size must match CUDA page size");
    }
    return config;
}

template <typename T>
ParameterViews<T> point_parameters(
    const T* base, const std::vector<std::size_t>& sizes) {
    if (sizes.size() != kParameterTensorCount) {
        throw std::logic_error("invalid GPT-2 parameter size table");
    }
    const T* cursor = base;
    auto take = [&](std::size_t index) {
        const T* result = cursor;
        cursor += sizes[index];
        return result;
    };
    ParameterViews<T> views;
    views.wte = take(0);
    views.wpe = take(1);
    views.ln1w = take(2);
    views.ln1b = take(3);
    views.qkvw = take(4);
    views.qkvb = take(5);
    views.attprojw = take(6);
    views.attprojb = take(7);
    views.ln2w = take(8);
    views.ln2b = take(9);
    views.fcw = take(10);
    views.fcb = take(11);
    views.fcprojw = take(12);
    views.fcprojb = take(13);
    views.lnfw = take(14);
    views.lnfb = take(15);
    return views;
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

__global__ void float_to_half_kernel(
    __half* output, const float* input, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(input[index]);
}

__global__ void float_to_bfloat16_kernel(
    __nv_bfloat16* output, const float* input, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2bfloat16_rn(input[index]);
}

template <typename T>
__global__ void embedding_kernel(
    T* output, const int* token_ids, const int* positions,
    const T* token_embeddings, const T* position_embeddings,
    int batch_size, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch_size * channels;
    if (index >= count) return;
    const int row = index / channels;
    const int channel = index % channels;
    output[index] = from_float<T>(
        to_float(token_embeddings[token_ids[row] * channels + channel]) +
        to_float(position_embeddings[positions[row] * channels + channel]));
}

template <typename T>
__global__ void layernorm_kernel(
    T* output, const T* input, const T* weight,
    const T* bias, int batch_size, int channels) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batch_size) return;
    __shared__ float reduction[kThreads];

    const T* row_input = input +
        static_cast<std::size_t>(row) * channels;
    float local_sum = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        local_sum += to_float(row_input[channel]);
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float mean = reduction[0] / channels;
    __syncthreads();

    float local_variance = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        const float shifted = to_float(row_input[channel]) - mean;
        local_variance += shifted * shifted;
    }
    reduction[thread] = local_variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float inverse_stddev =
        rsqrtf(reduction[0] / channels + 1e-5f);
    __syncthreads();

    T* row_output = output +
        static_cast<std::size_t>(row) * channels;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        row_output[channel] = from_float<T>(
            (to_float(row_input[channel]) - mean) * inverse_stddev *
                to_float(weight[channel]) +
            to_float(bias[channel]));
    }
}

// 将 Residual Add 与紧随其后的 LayerNorm 合并为一次 Launch。Residual 先按
// 目标存储类型舍入，再转回 FP32 参与归约，从而与未融合路径保持相同数值边界。
template <typename T>
__global__ void residual_layernorm_kernel(
    T* residual_output, T* normalized_output,
    const T* left, const T* right, const T* weight, const T* bias,
    int batch_size, int channels) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batch_size) return;
    __shared__ float reduction[kThreads];

    const std::size_t row_base =
        static_cast<std::size_t>(row) * channels;
    float local_sum = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        const std::size_t index = row_base + channel;
        const T residual = from_float<T>(
            to_float(left[index]) + to_float(right[index]));
        residual_output[index] = residual;
        local_sum += to_float(residual);
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float mean = reduction[0] / channels;
    __syncthreads();

    float local_variance = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        const float shifted =
            to_float(residual_output[row_base + channel]) - mean;
        local_variance += shifted * shifted;
    }
    reduction[thread] = local_variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float inverse_stddev =
        rsqrtf(reduction[0] / channels + 1e-5f);
    __syncthreads();

    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        const std::size_t index = row_base + channel;
        normalized_output[index] = from_float<T>(
            (to_float(residual_output[index]) - mean) * inverse_stddev *
                to_float(weight[channel]) +
            to_float(bias[channel]));
    }
}

__global__ void residual_layernorm_half2_kernel(
    __half* residual_output, __half* normalized_output,
    const __half* left, const __half* right,
    const __half* weight, const __half* bias,
    int batch_size, int channels) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batch_size) return;
    __shared__ float reduction[kThreads];

    const int pair_width = channels / 2;
    const std::size_t row_base =
        static_cast<std::size_t>(row) * pair_width;
    __half2* residual2 = reinterpret_cast<__half2*>(residual_output);
    __half2* normalized2 = reinterpret_cast<__half2*>(normalized_output);
    const __half2* left2 = reinterpret_cast<const __half2*>(left);
    const __half2* right2 = reinterpret_cast<const __half2*>(right);
    const __half2* weight2 = reinterpret_cast<const __half2*>(weight);
    const __half2* bias2 = reinterpret_cast<const __half2*>(bias);

    float local_sum = 0.0f;
    for (int pair = thread; pair < pair_width; pair += blockDim.x) {
        const std::size_t index = row_base + pair;
        const __half2 residual = __hadd2(left2[index], right2[index]);
        residual2[index] = residual;
        const float2 values = __half22float2(residual);
        local_sum += values.x + values.y;
    }
    reduction[thread] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float mean = reduction[0] / channels;
    __syncthreads();

    float local_variance = 0.0f;
    for (int pair = thread; pair < pair_width; pair += blockDim.x) {
        const float2 values = __half22float2(residual2[row_base + pair]);
        const float shifted_x = values.x - mean;
        const float shifted_y = values.y - mean;
        local_variance += shifted_x * shifted_x + shifted_y * shifted_y;
    }
    reduction[thread] = local_variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float inverse_stddev =
        rsqrtf(reduction[0] / channels + 1e-5f);
    __syncthreads();

    for (int pair = thread; pair < pair_width; pair += blockDim.x) {
        const std::size_t index = row_base + pair;
        const float2 values = __half22float2(residual2[index]);
        const float2 weights = __half22float2(weight2[pair]);
        const float2 biases = __half22float2(bias2[pair]);
        normalized2[index] = __floats2half2_rn(
            (values.x - mean) * inverse_stddev * weights.x + biases.x,
            (values.y - mean) * inverse_stddev * weights.y + biases.y);
    }
}

template <typename T>
__global__ void add_bias_kernel(
    T* output, const T* bias, int batch_size, int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch_size * width;
    if (index < count) {
        output[index] = from_float<T>(
            to_float(output[index]) + to_float(bias[index % width]));
    }
}

__global__ void add_bias_half2_kernel(
    __half* output, const __half* bias, int batch_size, int width) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_width = width / 2;
    const int count = batch_size * pair_width;
    if (pair < count) {
        reinterpret_cast<__half2*>(output)[pair] = __hadd2(
            reinterpret_cast<__half2*>(output)[pair],
            reinterpret_cast<const __half2*>(bias)[pair % pair_width]);
    }
}

template <typename T>
__global__ void split_qkv_kernel(
    const T* qkv, T* query, T* key, T* value,
    int batch_size, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch_size * channels;
    if (index >= count) return;
    const int row = index / channels;
    const int channel = index % channels;
    const std::size_t source =
        static_cast<std::size_t>(row) * 3 * channels + channel;
    query[index] = qkv[source];
    key[index] = qkv[source + channels];
    value[index] = qkv[source + 2 * channels];
}

template <typename T>
__global__ void residual_kernel(
    T* output, const T* left, const T* right, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] =
            from_float<T>(to_float(left[index]) + to_float(right[index]));
    }
}

__global__ void residual_half2_kernel(
    __half* output, const __half* left, const __half* right, int count) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair < count / 2) {
        reinterpret_cast<__half2*>(output)[pair] = __hadd2(
            reinterpret_cast<const __half2*>(left)[pair],
            reinterpret_cast<const __half2*>(right)[pair]);
    }
}

template <typename T>
__global__ void gelu_kernel(T* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = to_float(values[index]);
    const float cube = 0.044715f * value * value * value;
    values[index] = from_float<T>(
        0.5f * value *
        (1.0f + tanhf(0.7978845608028654f * (value + cube))));
}

__global__ void gelu_half2_kernel(__half* values, int count) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= count / 2) return;
    const float2 input =
        __half22float2(reinterpret_cast<__half2*>(values)[pair]);
    const float cube_x = 0.044715f * input.x * input.x * input.x;
    const float cube_y = 0.044715f * input.y * input.y * input.y;
    const float output_x = 0.5f * input.x *
        (1.0f + tanhf(0.7978845608028654f * (input.x + cube_x)));
    const float output_y = 0.5f * input.y *
        (1.0f + tanhf(0.7978845608028654f * (input.y + cube_y)));
    reinterpret_cast<__half2*>(values)[pair] =
        __floats2half2_rn(output_x, output_y);
}

__global__ void argmax_kernel(
    const float* logits, int* token_ids, int batch_size,
    int vocab_size, int padded_vocab_size) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batch_size) return;
    __shared__ float values[kThreads];
    __shared__ int indices[kThreads];

    float best_value = -FLT_MAX;
    int best_index = 0;
    const float* row_logits = logits +
        static_cast<std::size_t>(row) * padded_vocab_size;
    for (int index = thread; index < vocab_size; index += blockDim.x) {
        const float value = row_logits[index];
        if (value > best_value ||
            (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[thread] = best_value;
    indices[thread] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            const float other_value = values[thread + stride];
            const int other_index = indices[thread + stride];
            if (other_value > values[thread] ||
                (other_value == values[thread] &&
                 other_index < indices[thread])) {
                values[thread] = other_value;
                indices[thread] = other_index;
            }
        }
        __syncthreads();
    }
    if (thread == 0) token_ids[row] = indices[0];
}

// Gather 只选中每个已完成输入请求的最后一行，避免 Prompt 全行 LM Head。
template <typename T>
__global__ void gather_sample_rows_kernel(
    T* output, const T* input, const int* rows, int count, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * channels) return;
    const int row = index / channels;
    const int channel = index % channels;
    output[index] = input[static_cast<std::size_t>(rows[row]) * channels + channel];
}

int blocks_for(int count) {
    return (count + kThreads - 1) / kThreads;
}

void check_last_kernel(const char* operation) {
    check_cuda(cudaGetLastError(), operation);
}

} // namespace

class GPT2CudaModelRunner::Impl {
public:
    Impl(
        GPT2CudaConfig config, const float* host_parameters,
        std::size_t num_parameters, BlockManager& block_manager,
        std::size_t max_num_sequences,
        std::size_t max_num_batched_tokens,
        std::size_t max_context_length)
        : config_(validate_runner_arguments(
              config, host_parameters, num_parameters, block_manager,
              max_num_sequences, max_num_batched_tokens,
              max_context_length)),
          block_manager_(block_manager),
          max_num_sequences_(model_input_checked_int(
              max_num_sequences, "too many active CUDA sequences")),
          max_num_tokens_(model_input_checked_int(
              max_num_batched_tokens,
              "too many batched CUDA tokens")),
          max_context_length_(model_input_checked_int(
              max_context_length, "CUDA context capacity is too large")),
          max_blocks_per_sequence_(
              (max_context_length_ + kPagedAttentionPageSize - 1) /
              kPagedAttentionPageSize),
          num_pages_(model_input_checked_int(
              block_manager.num_blocks(), "too many CUDA KV pages")),
          parameter_sizes_(parameter_sizes(config_)),
          num_parameters_(sum(parameter_sizes_)),
          stream_(),
          cublas_(stream_.get(), config_.data_type),
          parameters_(num_parameters_, storage_size()),
          token_ids_(max_num_tokens_),
          positions_(max_num_tokens_),
          context_lengths_(max_num_tokens_),
          slot_mapping_(max_num_tokens_),
          block_tables_(
              static_cast<std::size_t>(max_num_tokens_) *
              max_blocks_per_sequence_),
          sampled_token_ids_(max_logit_rows()),
          sample_rows_(max_logit_rows()),
          key_cache_(cache_elements(), storage_size()),
          value_cache_(cache_elements(), storage_size()),
          residual_a_(batch_channels(), storage_size()),
          residual_b_(batch_channels(), storage_size()),
          normalized_(batch_channels(), storage_size()),
          qkv_(static_cast<std::size_t>(max_num_tokens_) *
               3 * config_.channels, storage_size()),
          query_(batch_channels(), storage_size()),
          key_(batch_channels(), storage_size()),
          value_(batch_channels(), storage_size()),
          attention_(batch_channels(), storage_size()),
          projected_(batch_channels(), storage_size()),
          hidden_(static_cast<std::size_t>(max_num_tokens_) *
                  4 * config_.channels, storage_size()),
          sampled_hidden_(static_cast<std::size_t>(max_logit_rows()) *
                          config_.channels, storage_size()),
          logits_(static_cast<std::size_t>(max_logit_rows()) *
                  config_.padded_vocab_size) {
        initialize_parameters(host_parameters);
        check_cuda(
            cudaMemsetAsync(
                key_cache_.data(), 0, key_cache_.bytes(), stream_.get()),
            "initialize key cache");
        check_cuda(
            cudaMemsetAsync(
                value_cache_.data(), 0, value_cache_.bytes(), stream_.get()),
            "initialize value cache");
        check_cuda(cudaStreamSynchronize(stream_.get()),
                   "finish CUDA ModelRunner initialization");
    }

    ~Impl() {
        for (auto& item : cuda_graphs_) {
            if (item.second.executable != nullptr) {
                cudaGraphExecDestroy(item.second.executable);
            }
            if (item.second.graph != nullptr) {
                cudaGraphDestroy(item.second.graph);
            }
        }
    }

    std::vector<int> run(const SchedulerOutput& output) {
        if (output.items.empty()) {
            throw std::invalid_argument(
                "CUDA ModelRunner received an empty schedule");
        }
        if (output.items.size() >
            static_cast<std::size_t>(max_num_sequences_)) {
            throw std::out_of_range(
                "scheduled request count exceeds CUDA runner capacity");
        }
        if (output.num_batched_tokens >
            static_cast<std::size_t>(max_num_tokens_)) {
            throw std::out_of_range(
                "scheduled token count exceeds CUDA runner capacity");
        }

        std::vector<int> sampled(output.items.size(), -1);
        last_model_inputs_.clear();
        last_host_to_device_bytes_ = 0;
        ModelInput input = prepare_packed_model_input(
            output, block_manager_, max_context_length_,
            max_blocks_per_sequence_, num_pages_);
        last_logit_token_indices_.clear();
        if (config_.enable_sample_row_pruning) {
            for (std::size_t i = 0; i < output.items.size(); ++i) {
                const auto& item = output.items[i];
                if (item.sequence->num_computed_tokens() + item.num_scheduled_tokens ==
                    item.sequence->num_tokens()) {
                    last_logit_token_indices_.push_back(static_cast<int>(
                        input.query_start_locations[i + 1] - 1));
                }
            }
        } else {
            last_logit_token_indices_.resize(input.batch_size());
            std::iota(last_logit_token_indices_.begin(), last_logit_token_indices_.end(), 0);
        }
        const std::vector<int> token_samples =
            config_.data_type == CudaDataType::FP16
                ? forward<__half>(input)
                : (config_.data_type == CudaDataType::BF16
                    ? forward<__nv_bfloat16>(input)
                    : forward<float>(input));
        std::size_t sample_index = 0;
        for (std::size_t item_index = 0;
             item_index < output.items.size(); ++item_index) {
            const ScheduledItem& item = output.items[item_index];
            const Sequence& sequence = *item.sequence;
            if (sequence.num_computed_tokens() +
                    item.num_scheduled_tokens ==
                sequence.num_tokens()) {
                const std::size_t final_token =
                    input.query_start_locations[item_index + 1] - 1;
                sampled[item_index] = token_samples[
                    config_.enable_sample_row_pruning ? sample_index++ : final_token];
            }
        }
        last_model_inputs_.push_back(std::move(input));
        return sampled;
    }

    std::vector<float> last_logits_for_testing() const {
        if (last_logit_token_indices_.empty()) return {};
        std::vector<float> result(
            last_logit_token_indices_.size() *
            config_.padded_vocab_size);
        check_cuda(
            cudaMemcpy(
                result.data(), logits_.get(),
                result.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "copy debug logits to host");
        return result;
    }

    const std::vector<int>& last_logit_token_indices() const {
        return last_logit_token_indices_;
    }

    const std::vector<ModelInput>& last_model_inputs() const {
        return last_model_inputs_;
    }
    std::size_t last_host_to_device_bytes() const {
        return last_host_to_device_bytes_;
    }
    std::size_t weight_bytes() const { return parameters_.bytes(); }
    std::size_t kv_cache_bytes() const {
        return key_cache_.bytes() + value_cache_.bytes();
    }
    std::size_t activation_bytes() const {
        return residual_a_.bytes() + residual_b_.bytes() +
            normalized_.bytes() + qkv_.bytes() + query_.bytes() +
            key_.bytes() + value_.bytes() + attention_.bytes() +
            projected_.bytes() + hidden_.bytes() + sampled_hidden_.bytes() + logits_.bytes();
    }
    CudaDataType data_type() const { return config_.data_type; }
    std::size_t num_cuda_graphs() const { return cuda_graphs_.size(); }

    KVTransferStats copy_kv_to(Impl& destination, const Sequence& source,
                              const Sequence& target, std::size_t computed_tokens) {
        if (this == &destination || config_.device_id == destination.config_.device_id ||
            config_.data_type != destination.config_.data_type ||
            config_.num_layers != destination.config_.num_layers ||
            config_.num_heads != destination.config_.num_heads ||
            config_.channels != destination.config_.channels ||
            computed_tokens == 0 || computed_tokens > source.num_computed_tokens() ||
            computed_tokens > target.num_tokens() ||
            computed_tokens > static_cast<std::size_t>(destination.max_context_length_)) {
            throw std::invalid_argument("incompatible KV handoff");
        }
        if (!std::equal(source.token_ids().begin(),
                        source.token_ids().begin() + computed_tokens,
                        target.token_ids().begin())) {
            throw std::invalid_argument("KV handoff token prefix mismatch");
        }
        KVTransferStats result;
        result.num_pages = block_manager_.blocks_needed(computed_tokens);
        if (source.block_table().size() < result.num_pages ||
            target.block_table().size() < result.num_pages) {
            throw std::invalid_argument("KV handoff requires allocated page tables");
        }
        // 在任何拷贝前检查所有页号，避免部分写入后才发现越界。
        for (std::size_t i = 0; i < result.num_pages; ++i) {
            if (source.block_table()[i] < 0 || source.block_table()[i] >= num_pages_ ||
                target.block_table()[i] < 0 || target.block_table()[i] >= destination.num_pages_) {
                throw std::out_of_range("KV handoff page out of range");
            }
        }
        const std::size_t page_bytes = key_cache_.bytes() / num_pages_;
        result.payload_bytes = 2 * result.num_pages * page_bytes;
        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const auto start = Clock::now();
        void* staging = nullptr;
        check_cuda(cudaHostAlloc(&staging, result.payload_bytes, cudaHostAllocPortable),
                   "allocate pinned KV staging");
        try {
            const auto begin_d2h = Clock::now();
            {
                DeviceGuard guard(config_.device_id);
                for (int kv = 0; kv < 2; ++kv) {
                    const auto* cache = static_cast<const char*>(
                        kv == 0 ? key_cache_.data() : value_cache_.data());
                    for (std::size_t i = 0; i < result.num_pages; ++i) {
                        check_cuda(cudaMemcpyAsync(
                            static_cast<char*>(staging) + (kv * result.num_pages + i) * page_bytes,
                            cache + source.block_table()[i] * page_bytes, page_bytes,
                            cudaMemcpyDeviceToHost, stream_.get()), "stage source KV page");
                    }
                }
                check_cuda(cudaStreamSynchronize(stream_.get()), "complete KV D2H");
            }
            const auto begin_h2d = Clock::now();
            {
                DeviceGuard guard(destination.config_.device_id);
                for (int kv = 0; kv < 2; ++kv) {
                    auto* cache = static_cast<char*>(kv == 0
                        ? destination.key_cache_.data() : destination.value_cache_.data());
                    for (std::size_t i = 0; i < result.num_pages; ++i) {
                        check_cuda(cudaMemcpyAsync(
                            cache + target.block_table()[i] * page_bytes,
                            static_cast<char*>(staging) + (kv * result.num_pages + i) * page_bytes,
                            page_bytes, cudaMemcpyHostToDevice, destination.stream_.get()),
                            "restore destination KV page");
                    }
                }
                check_cuda(cudaStreamSynchronize(destination.stream_.get()), "complete KV H2D");
            }
            result.d2h_ms = ms(begin_d2h, begin_h2d);
            result.h2d_ms = ms(begin_h2d, Clock::now());
        } catch (...) {
            // 异常路径也要等未完成 DMA 结束才能释放 staging。
            { DeviceGuard guard(config_.device_id); cudaStreamSynchronize(stream_.get()); }
            { DeviceGuard guard(destination.config_.device_id); cudaStreamSynchronize(destination.stream_.get()); }
            cudaFreeHost(staging);
            throw;
        }
        check_cuda(cudaFreeHost(staging), "free pinned KV staging");
        result.total_ms = ms(start, Clock::now());
        return result;
    }

private:
    int max_logit_rows() const {
        return config_.enable_sample_row_pruning ? max_num_sequences_ : max_num_tokens_;
    }

    static std::size_t sum(const std::vector<std::size_t>& values) {
        std::size_t result = 0;
        for (std::size_t value : values) result += value;
        return result;
    }

    std::size_t batch_channels() const {
        return static_cast<std::size_t>(max_num_tokens_) * config_.channels;
    }

    std::size_t storage_size() const {
        return storage_element_size(config_.data_type);
    }

    void initialize_parameters(const float* host_parameters) {
        if (config_.data_type == CudaDataType::FP32) {
            check_cuda(
                cudaMemcpyAsync(
                    parameters_.get<float>(), host_parameters,
                    parameters_.bytes(), cudaMemcpyHostToDevice,
                    stream_.get()),
                "copy FP32 GPT-2 weights to GPU");
            return;
        }

        DeviceBuffer<float> fp32_parameters(num_parameters_);
        check_cuda(
            cudaMemcpyAsync(
                fp32_parameters.get(), host_parameters,
                fp32_parameters.bytes(), cudaMemcpyHostToDevice,
                stream_.get()),
            "copy GPT-2 weights before FP16 conversion");
        if (config_.data_type == CudaDataType::FP16) {
            float_to_half_kernel<<<
                blocks_for(static_cast<int>(num_parameters_)), kThreads, 0,
                stream_.get()>>>(
                parameters_.get<__half>(), fp32_parameters.get(),
                num_parameters_);
            check_last_kernel("convert GPT-2 weights to FP16");
        } else {
            float_to_bfloat16_kernel<<<
                blocks_for(static_cast<int>(num_parameters_)), kThreads, 0,
                stream_.get()>>>(
                parameters_.get<__nv_bfloat16>(), fp32_parameters.get(),
                num_parameters_);
            check_last_kernel("convert GPT-2 weights to BF16");
        }
        check_cuda(
            cudaStreamSynchronize(stream_.get()),
            "finish reduced-precision weight conversion");
    }

    std::size_t cache_elements() const {
        return static_cast<std::size_t>(num_pages_) * config_.num_layers *
            config_.num_heads * kPagedAttentionPageSize *
            (config_.channels / config_.num_heads);
    }

    template <typename T>
    void matmul(
        T* output, const T* input, const T* weight,
        const T* bias, int batch_size, int input_width,
        int output_width) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if constexpr (std::is_same<T, float>::value) {
            check_cublas(cublasSgemm(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, input_width, input, input_width, &beta,
                output, output_width), "cublasSgemm");
        } else {
            constexpr cudaDataType_t storage_type =
                std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
            check_cublas(cublasGemmEx(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, storage_type, input_width,
                input, storage_type, input_width, &beta,
                output, storage_type, output_width,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx FP16 Tensor Core");
        }
        if (bias != nullptr) {
            if constexpr (std::is_same<T, __half>::value) {
                add_bias_half2_kernel<<<
                    blocks_for(batch_size * output_width / 2), kThreads, 0,
                    stream_.get()>>>(
                    output, bias, batch_size, output_width);
            } else {
                add_bias_kernel<T><<<
                    blocks_for(batch_size * output_width), kThreads, 0,
                    stream_.get()>>>(
                    output, bias, batch_size, output_width);
            }
            check_last_kernel("add_bias_kernel");
        }
    }

    template <typename T>
    void logits_matmul(
        const T* input, const T* weight, int batch_size, int input_width,
        int output_width) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if constexpr (std::is_same<T, float>::value) {
            check_cublas(cublasSgemm(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, input_width, input, input_width, &beta,
                logits_.get(), output_width), "cublasSgemm logits");
        } else {
            constexpr cudaDataType_t storage_type =
                std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
            check_cublas(cublasGemmEx(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, storage_type, input_width,
                input, storage_type, input_width, &beta,
                logits_.get(), CUDA_R_32F, output_width,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx FP16 logits");
        }
    }

    template <typename T>
    void fused_residual_layernorm(
        T* residual_output, T* normalized_output,
        const T* left, const T* right, const T* weight, const T* bias,
        int batch_size, int channels) {
        if constexpr (std::is_same<T, __half>::value) {
            if (channels % 2 == 0) {
                residual_layernorm_half2_kernel<<<
                    batch_size, kThreads, 0, stream_.get()>>>(
                    residual_output, normalized_output, left, right,
                    weight, bias, batch_size, channels);
            } else {
                residual_layernorm_kernel<T><<<
                    batch_size, kThreads, 0, stream_.get()>>>(
                    residual_output, normalized_output, left, right,
                    weight, bias, batch_size, channels);
            }
        } else {
            residual_layernorm_kernel<T><<<
                batch_size, kThreads, 0, stream_.get()>>>(
                residual_output, normalized_output, left, right,
                weight, bias, batch_size, channels);
        }
        check_last_kernel("fused residual layernorm");
    }

    template <typename T>
    std::vector<int> forward(const ModelInput& input) {
        const int batch_size = model_input_checked_int(
            input.batch_size(), "CUDA packed batch is too large");
        if (batch_size <= 0 || batch_size > max_num_tokens_ ||
            input.positions.size() != input.batch_size() ||
            input.context_lengths.size() != input.batch_size() ||
            input.slot_mapping.size() != input.batch_size() ||
            input.block_tables.size() !=
                input.batch_size() * max_blocks_per_sequence_) {
            throw std::invalid_argument("invalid CUDA ModelInput shape");
        }

        copy_metadata(token_ids_, input.token_ids, "copy token ids");
        copy_metadata(positions_, input.positions, "copy positions");
        copy_metadata(
            context_lengths_, input.context_lengths,
            "copy context lengths");
        copy_metadata(
            slot_mapping_, input.slot_mapping, "copy slot mapping");
        copy_metadata(
            block_tables_, input.block_tables, "copy block tables");

        const int num_logit_rows = static_cast<int>(last_logit_token_indices_.size());
        if (config_.enable_sample_row_pruning && num_logit_rows > 0) {
            copy_metadata(sample_rows_, last_logit_token_indices_, "copy sample rows");
        }
        // Grid/GEMM 随输入行数和采样行数变化；行索引本身在图外更新。
        const auto graph_key = std::make_pair(batch_size, num_logit_rows);
        bool replay_existing_graph = false;
        bool capture_new_graph = false;
        if (config_.enable_cuda_graph) {
            const auto graph = cuda_graphs_.find(graph_key);
            if (graph != cuda_graphs_.end()) {
                check_cuda(
                    cudaGraphLaunch(graph->second.executable, stream_.get()),
                    "launch cached CUDA graph");
                replay_existing_graph = true;
            } else {
                // Stream capture cannot begin behind uncaptured metadata copies.
                // The first use of each Token Batch therefore synchronizes once;
                // subsequent replays keep metadata and graph launch ordered on
                // the same stream without this synchronization.
                check_cuda(
                    cudaStreamSynchronize(stream_.get()),
                    "prepare CUDA graph capture");
                check_cuda(
                    cudaStreamBeginCapture(
                        stream_.get(), cudaStreamCaptureModeThreadLocal),
                    "begin CUDA graph capture");
                capture_new_graph = true;
            }
        }

        if (!replay_existing_graph) {
        const int channels = config_.channels;
        const int hidden_width = 4 * channels;
        const int channel_elements = batch_size * channels;
        const ParameterViews<T> parameters_view =
            point_parameters(parameters_.get<T>(), parameter_sizes_);
        embedding_kernel<T><<<
            blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
            residual_a_.get<T>(), token_ids_.get(), positions_.get(),
            parameters_view.wte, parameters_view.wpe,
            batch_size, channels);
        check_last_kernel("embedding_kernel");

        if (config_.enable_fused_residual_layernorm) {
            layernorm_kernel<T><<<batch_size, kThreads, 0, stream_.get()>>>(
                normalized_.get<T>(), residual_a_.get<T>(),
                parameters_view.ln1w, parameters_view.ln1b,
                batch_size, channels);
            check_last_kernel("initial layernorm ln1");
        }

        for (int layer = 0; layer < config_.num_layers; ++layer) {
            if (!config_.enable_fused_residual_layernorm) {
                layernorm_kernel<T><<<
                    batch_size, kThreads, 0, stream_.get()>>>(
                    normalized_.get<T>(), residual_a_.get<T>(),
                    parameters_view.ln1w +
                        static_cast<std::size_t>(layer) * channels,
                    parameters_view.ln1b +
                        static_cast<std::size_t>(layer) * channels,
                    batch_size, channels);
                check_last_kernel("layernorm ln1");
            }

            matmul(
                qkv_.get<T>(), normalized_.get<T>(),
                parameters_view.qkvw +
                    static_cast<std::size_t>(layer) * 3 * channels * channels,
                parameters_view.qkvb +
                    static_cast<std::size_t>(layer) * 3 * channels,
                batch_size, channels, 3 * channels);
            split_qkv_kernel<T><<<
                blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
                qkv_.get<T>(), query_.get<T>(), key_.get<T>(), value_.get<T>(),
                batch_size, channels);
            check_last_kernel("split_qkv_kernel");

            check_cuda(
                paged_attention_decode(
                    query_.get<T>(), key_.get<T>(), value_.get<T>(),
                    key_cache_.get<T>(), value_cache_.get<T>(),
                    block_tables_.get(), context_lengths_.get(),
                    slot_mapping_.get(), attention_.get<T>(), batch_size,
                    num_pages_, config_.num_layers, layer,
                    config_.num_heads, channels / config_.num_heads,
                    max_blocks_per_sequence_, max_context_length_,
                    stream_.get()),
                "paged_attention_decode");

            matmul(
                projected_.get<T>(), attention_.get<T>(),
                parameters_view.attprojw +
                    static_cast<std::size_t>(layer) * channels * channels,
                parameters_view.attprojb +
                    static_cast<std::size_t>(layer) * channels,
                batch_size, channels, channels);
            if (config_.enable_fused_residual_layernorm) {
                fused_residual_layernorm(
                    residual_b_.get<T>(), normalized_.get<T>(),
                    residual_a_.get<T>(), projected_.get<T>(),
                    parameters_view.ln2w +
                        static_cast<std::size_t>(layer) * channels,
                    parameters_view.ln2b +
                        static_cast<std::size_t>(layer) * channels,
                    batch_size, channels);
            } else {
                if constexpr (std::is_same<T, __half>::value) {
                    residual_half2_kernel<<<
                        blocks_for(channel_elements / 2), kThreads, 0,
                        stream_.get()>>>(
                        residual_b_.get<T>(), residual_a_.get<T>(),
                        projected_.get<T>(), channel_elements);
                } else {
                    residual_kernel<T><<<
                        blocks_for(channel_elements), kThreads, 0,
                        stream_.get()>>>(
                        residual_b_.get<T>(), residual_a_.get<T>(),
                        projected_.get<T>(), channel_elements);
                }
            }
            check_last_kernel("attention residual");

            if (!config_.enable_fused_residual_layernorm) {
                layernorm_kernel<T><<<
                    batch_size, kThreads, 0, stream_.get()>>>(
                    normalized_.get<T>(), residual_b_.get<T>(),
                    parameters_view.ln2w +
                        static_cast<std::size_t>(layer) * channels,
                    parameters_view.ln2b +
                        static_cast<std::size_t>(layer) * channels,
                    batch_size, channels);
                check_last_kernel("layernorm ln2");
            }
            matmul(
                hidden_.get<T>(), normalized_.get<T>(),
                parameters_view.fcw +
                    static_cast<std::size_t>(layer) *
                    hidden_width * channels,
                parameters_view.fcb +
                    static_cast<std::size_t>(layer) * hidden_width,
                batch_size, channels, hidden_width);
            if constexpr (std::is_same<T, __half>::value) {
                gelu_half2_kernel<<<
                    blocks_for(batch_size * hidden_width / 2), kThreads, 0,
                    stream_.get()>>>(
                    hidden_.get<T>(), batch_size * hidden_width);
            } else {
                gelu_kernel<T><<<
                    blocks_for(batch_size * hidden_width), kThreads, 0,
                    stream_.get()>>>(
                    hidden_.get<T>(), batch_size * hidden_width);
            }
            check_last_kernel("gelu_kernel");
            matmul(
                projected_.get<T>(), hidden_.get<T>(),
                parameters_view.fcprojw +
                    static_cast<std::size_t>(layer) *
                    channels * hidden_width,
                parameters_view.fcprojb +
                    static_cast<std::size_t>(layer) * channels,
                batch_size, hidden_width, channels);
            if (config_.enable_fused_residual_layernorm) {
                const bool has_next_layer = layer + 1 < config_.num_layers;
                const T* norm_weight = has_next_layer
                    ? parameters_view.ln1w +
                        static_cast<std::size_t>(layer + 1) * channels
                    : parameters_view.lnfw;
                const T* norm_bias = has_next_layer
                    ? parameters_view.ln1b +
                        static_cast<std::size_t>(layer + 1) * channels
                    : parameters_view.lnfb;
                fused_residual_layernorm(
                    residual_a_.get<T>(), normalized_.get<T>(),
                    residual_b_.get<T>(), projected_.get<T>(),
                    norm_weight, norm_bias, batch_size, channels);
            } else {
                if constexpr (std::is_same<T, __half>::value) {
                    residual_half2_kernel<<<
                        blocks_for(channel_elements / 2), kThreads, 0,
                        stream_.get()>>>(
                        residual_a_.get<T>(), residual_b_.get<T>(),
                        projected_.get<T>(), channel_elements);
                } else {
                    residual_kernel<T><<<
                        blocks_for(channel_elements), kThreads, 0,
                        stream_.get()>>>(
                        residual_a_.get<T>(), residual_b_.get<T>(),
                        projected_.get<T>(), channel_elements);
                }
            }
            check_last_kernel("MLP residual");
        }

        if (!config_.enable_fused_residual_layernorm) {
            layernorm_kernel<T><<<batch_size, kThreads, 0, stream_.get()>>>(
                normalized_.get<T>(), residual_a_.get<T>(),
                parameters_view.lnfw, parameters_view.lnfb,
                batch_size, channels);
            check_last_kernel("final layernorm");
        }
        if (num_logit_rows > 0) {
            const T* lm_input = normalized_.get<T>();
            if (config_.enable_sample_row_pruning) {
                gather_sample_rows_kernel<T><<<
                    blocks_for(num_logit_rows * channels), kThreads, 0, stream_.get()>>>(
                    sampled_hidden_.get<T>(), normalized_.get<T>(), sample_rows_.get(),
                    num_logit_rows, channels);
                check_last_kernel("gather_sample_rows_kernel");
                lm_input = sampled_hidden_.get<T>();
            }
            logits_matmul(lm_input, parameters_view.wte,
                          num_logit_rows, channels, config_.padded_vocab_size);
            argmax_kernel<<<num_logit_rows, kThreads, 0, stream_.get()>>>(
                logits_.get(), sampled_token_ids_.get(), num_logit_rows,
                config_.vocab_size, config_.padded_vocab_size);
            check_last_kernel("argmax_kernel");
        }
        }

        if (capture_new_graph) {
            CudaGraphEntry entry;
            check_cuda(
                cudaStreamEndCapture(stream_.get(), &entry.graph),
                "end CUDA graph capture");
            check_cuda(
                cudaGraphInstantiate(
                    &entry.executable, entry.graph, nullptr, nullptr, 0),
                "instantiate CUDA graph");
            const auto inserted =
                cuda_graphs_.emplace(graph_key, entry);
            if (!inserted.second) {
                cudaGraphExecDestroy(entry.executable);
                cudaGraphDestroy(entry.graph);
                throw std::logic_error("duplicate CUDA graph batch key");
            }
            check_cuda(
                cudaGraphLaunch(entry.executable, stream_.get()),
                "launch newly captured CUDA graph");
        }

        std::vector<int> sampled(num_logit_rows);
        if (num_logit_rows > 0) check_cuda(
            cudaMemcpyAsync(
                sampled.data(), sampled_token_ids_.get(),
                sampled.size() * sizeof(int), cudaMemcpyDeviceToHost,
                stream_.get()),
            "copy sampled token ids");
        check_cuda(
            cudaStreamSynchronize(stream_.get()),
            "finish CUDA GPT-2 micro batch");
        return sampled;
    }

    void copy_metadata(
        DeviceBuffer<int>& destination, const std::vector<int>& source,
        const char* operation) {
        if (source.size() > destination.count()) {
            throw std::out_of_range("CUDA metadata exceeds buffer capacity");
        }
        const std::size_t bytes = source.size() * sizeof(int);
        check_cuda(
            cudaMemcpyAsync(
                destination.get(), source.data(), bytes,
                cudaMemcpyHostToDevice, stream_.get()),
            operation);
        last_host_to_device_bytes_ += bytes;
    }

    GPT2CudaConfig config_;
    BlockManager& block_manager_;
    int max_num_sequences_;
    int max_num_tokens_;
    int max_context_length_;
    int max_blocks_per_sequence_;
    int num_pages_;
    std::vector<std::size_t> parameter_sizes_;
    std::size_t num_parameters_;
    CudaStream stream_;
    CublasHandle cublas_;
    DeviceTensorBuffer parameters_;
    DeviceBuffer<int> token_ids_;
    DeviceBuffer<int> positions_;
    DeviceBuffer<int> context_lengths_;
    DeviceBuffer<int> slot_mapping_;
    DeviceBuffer<int> block_tables_;
    DeviceBuffer<int> sampled_token_ids_;
    DeviceBuffer<int> sample_rows_;
    DeviceTensorBuffer key_cache_;
    DeviceTensorBuffer value_cache_;
    DeviceTensorBuffer residual_a_;
    DeviceTensorBuffer residual_b_;
    DeviceTensorBuffer normalized_;
    DeviceTensorBuffer qkv_;
    DeviceTensorBuffer query_;
    DeviceTensorBuffer key_;
    DeviceTensorBuffer value_;
    DeviceTensorBuffer attention_;
    DeviceTensorBuffer projected_;
    DeviceTensorBuffer hidden_;
    DeviceTensorBuffer sampled_hidden_;
    DeviceBuffer<float> logits_;
    std::vector<int> last_logit_token_indices_;
    std::vector<ModelInput> last_model_inputs_;
    std::size_t last_host_to_device_bytes_ = 0;

    struct CudaGraphEntry {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
    };
    std::map<std::pair<int, int>, CudaGraphEntry> cuda_graphs_;
};

GPT2CudaModelRunner::GPT2CudaModelRunner(
    GPT2CudaConfig config, const float* host_parameters,
    std::size_t num_parameters, BlockManager& block_manager,
    std::size_t max_num_sequences,
    std::size_t max_num_batched_tokens,
    std::size_t max_context_length)
    : device_id_(config.device_id) {
    if (device_id_ < 0) check_cuda(cudaGetDevice(&device_id_), "resolve runner device");
    DeviceGuard guard(device_id_);
    config.device_id = device_id_;
    impl_ = std::make_unique<Impl>(
        config, host_parameters, num_parameters, block_manager,
        max_num_sequences, max_num_batched_tokens, max_context_length);
}

GPT2CudaModelRunner::~GPT2CudaModelRunner() {
    // 析构不抛异常，但必须在所属设备销毁图、缓冲区、cuBLAS 和 Stream。
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device_id_);
    impl_.reset();
    cudaSetDevice(previous);
}

std::vector<int> GPT2CudaModelRunner::run(const SchedulerOutput& output) {
    DeviceGuard guard(device_id_);
    return impl_->run(output);
}

KVTransferStats GPT2CudaModelRunner::copy_kv_to(
    GPT2CudaModelRunner& destination, const Sequence& source,
    const Sequence& target, std::size_t computed_tokens) {
    DeviceGuard guard(device_id_);
    return impl_->copy_kv_to(*destination.impl_, source, target, computed_tokens);
}

const std::vector<ModelInput>& GPT2CudaModelRunner::last_model_inputs() const {
    return impl_->last_model_inputs();
}

std::vector<float> GPT2CudaModelRunner::last_logits_for_testing() const {
    DeviceGuard guard(device_id_);
    return impl_->last_logits_for_testing();
}

const std::vector<int>& GPT2CudaModelRunner::last_logit_token_indices() const {
    return impl_->last_logit_token_indices();
}

std::size_t GPT2CudaModelRunner::last_host_to_device_bytes() const {
    return impl_->last_host_to_device_bytes();
}

std::size_t GPT2CudaModelRunner::weight_bytes() const {
    return impl_->weight_bytes();
}

std::size_t GPT2CudaModelRunner::kv_cache_bytes() const {
    return impl_->kv_cache_bytes();
}

std::size_t GPT2CudaModelRunner::activation_bytes() const {
    return impl_->activation_bytes();
}

CudaDataType GPT2CudaModelRunner::data_type() const {
    return impl_->data_type();
}

std::size_t GPT2CudaModelRunner::num_cuda_graphs() const {
    return impl_->num_cuda_graphs();
}

} // namespace cuda
} // namespace mini_vllm
