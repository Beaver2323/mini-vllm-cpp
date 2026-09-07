#include "gpt2_cuda_model_runner.cuh"
#include "paged_attention.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_vllm {
namespace cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kParameterTensorCount = 16;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

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
    explicit CublasHandle(cudaStream_t stream) {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
        check_cublas(
            cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode");
    }
    ~CublasHandle() {
        if (handle_ != nullptr) cublasDestroy(handle_);
    }
    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

struct ParameterViews {
    const float* wte = nullptr;
    const float* wpe = nullptr;
    const float* ln1w = nullptr;
    const float* ln1b = nullptr;
    const float* qkvw = nullptr;
    const float* qkvb = nullptr;
    const float* attprojw = nullptr;
    const float* attprojb = nullptr;
    const float* ln2w = nullptr;
    const float* ln2b = nullptr;
    const float* fcw = nullptr;
    const float* fcb = nullptr;
    const float* fcprojw = nullptr;
    const float* fcprojb = nullptr;
    const float* lnfw = nullptr;
    const float* lnfb = nullptr;
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

ParameterViews point_parameters(
    const float* base, const std::vector<std::size_t>& sizes) {
    if (sizes.size() != kParameterTensorCount) {
        throw std::logic_error("invalid GPT-2 parameter size table");
    }
    const float* cursor = base;
    auto take = [&](std::size_t index) {
        const float* result = cursor;
        cursor += sizes[index];
        return result;
    };
    ParameterViews views;
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

__global__ void embedding_kernel(
    float* output, const int* token_ids, const int* positions,
    const float* token_embeddings, const float* position_embeddings,
    int batch_size, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch_size * channels;
    if (index >= count) return;
    const int row = index / channels;
    const int channel = index % channels;
    output[index] =
        token_embeddings[token_ids[row] * channels + channel] +
        position_embeddings[positions[row] * channels + channel];
}

__global__ void layernorm_kernel(
    float* output, const float* input, const float* weight,
    const float* bias, int batch_size, int channels) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batch_size) return;
    __shared__ float reduction[kThreads];

    const float* row_input = input +
        static_cast<std::size_t>(row) * channels;
    float local_sum = 0.0f;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        local_sum += row_input[channel];
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
        const float shifted = row_input[channel] - mean;
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

    float* row_output = output +
        static_cast<std::size_t>(row) * channels;
    for (int channel = thread; channel < channels;
         channel += blockDim.x) {
        row_output[channel] =
            (row_input[channel] - mean) * inverse_stddev * weight[channel] +
            bias[channel];
    }
}

__global__ void add_bias_kernel(
    float* output, const float* bias, int batch_size, int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch_size * width;
    if (index < count) output[index] += bias[index % width];
}

__global__ void split_qkv_kernel(
    const float* qkv, float* query, float* key, float* value,
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

__global__ void residual_kernel(
    float* output, const float* left, const float* right, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = left[index] + right[index];
}

__global__ void gelu_kernel(float* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = values[index];
    const float cube = 0.044715f * value * value * value;
    values[index] =
        0.5f * value *
        (1.0f + tanhf(0.7978845608028654f * (value + cube)));
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
          cublas_(stream_.get()),
          parameters_(num_parameters_),
          token_ids_(max_num_tokens_),
          positions_(max_num_tokens_),
          context_lengths_(max_num_tokens_),
          slot_mapping_(max_num_tokens_),
          block_tables_(
              static_cast<std::size_t>(max_num_tokens_) *
              max_blocks_per_sequence_),
          sampled_token_ids_(max_num_tokens_),
          key_cache_(cache_elements()),
          value_cache_(cache_elements()),
          residual_a_(batch_channels()),
          residual_b_(batch_channels()),
          normalized_(batch_channels()),
          qkv_(static_cast<std::size_t>(max_num_tokens_) *
               3 * config_.channels),
          query_(batch_channels()),
          key_(batch_channels()),
          value_(batch_channels()),
          attention_(batch_channels()),
          projected_(batch_channels()),
          hidden_(static_cast<std::size_t>(max_num_tokens_) *
                  4 * config_.channels),
          logits_(static_cast<std::size_t>(max_num_tokens_) *
                  config_.padded_vocab_size) {
        check_cuda(
            cudaMemcpyAsync(
                parameters_.get(), host_parameters, parameters_.bytes(),
                cudaMemcpyHostToDevice, stream_.get()),
            "copy GPT-2 weights to GPU");
        check_cuda(
            cudaMemsetAsync(
                key_cache_.get(), 0, key_cache_.bytes(), stream_.get()),
            "initialize key cache");
        check_cuda(
            cudaMemsetAsync(
                value_cache_.get(), 0, value_cache_.bytes(), stream_.get()),
            "initialize value cache");
        check_cuda(cudaStreamSynchronize(stream_.get()),
                   "finish CUDA ModelRunner initialization");
        parameters_view_ =
            point_parameters(parameters_.get(), parameter_sizes_);
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
        const std::vector<int> token_samples = forward(input);
        for (std::size_t item_index = 0;
             item_index < output.items.size(); ++item_index) {
            const ScheduledItem& item = output.items[item_index];
            const Sequence& sequence = *item.sequence;
            if (sequence.num_computed_tokens() +
                    item.num_scheduled_tokens ==
                sequence.num_tokens()) {
                const std::size_t final_token =
                    input.query_start_locations[item_index + 1] - 1;
                sampled[item_index] = token_samples[final_token];
            }
        }
        last_model_inputs_.push_back(std::move(input));
        return sampled;
    }

    std::vector<float> last_logits_for_testing() const {
        if (last_batch_size_ == 0) return {};
        std::vector<float> result(
            static_cast<std::size_t>(last_batch_size_) *
            config_.padded_vocab_size);
        check_cuda(
            cudaMemcpy(
                result.data(), logits_.get(),
                result.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "copy debug logits to host");
        return result;
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
            projected_.bytes() + hidden_.bytes() + logits_.bytes();
    }

private:
    static std::size_t sum(const std::vector<std::size_t>& values) {
        std::size_t result = 0;
        for (std::size_t value : values) result += value;
        return result;
    }

    std::size_t batch_channels() const {
        return static_cast<std::size_t>(max_num_tokens_) * config_.channels;
    }

    std::size_t cache_elements() const {
        return static_cast<std::size_t>(num_pages_) * config_.num_layers *
            config_.num_heads * kPagedAttentionPageSize *
            (config_.channels / config_.num_heads);
    }

    void matmul(
        float* output, const float* input, const float* weight,
        const float* bias, int batch_size, int input_width,
        int output_width) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        check_cublas(
            cublasSgemm(
                cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                output_width, batch_size, input_width, &alpha,
                weight, input_width, input, input_width, &beta,
                output, output_width),
            "cublasSgemm");
        if (bias != nullptr) {
            add_bias_kernel<<<
                blocks_for(batch_size * output_width), kThreads, 0,
                stream_.get()>>>(output, bias, batch_size, output_width);
            check_last_kernel("add_bias_kernel");
        }
    }

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

        const int channels = config_.channels;
        const int hidden_width = 4 * channels;
        const int channel_elements = batch_size * channels;
        embedding_kernel<<<
            blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
            residual_a_.get(), token_ids_.get(), positions_.get(),
            parameters_view_.wte, parameters_view_.wpe,
            batch_size, channels);
        check_last_kernel("embedding_kernel");

        for (int layer = 0; layer < config_.num_layers; ++layer) {
            layernorm_kernel<<<batch_size, kThreads, 0, stream_.get()>>>(
                normalized_.get(), residual_a_.get(),
                parameters_view_.ln1w +
                    static_cast<std::size_t>(layer) * channels,
                parameters_view_.ln1b +
                    static_cast<std::size_t>(layer) * channels,
                batch_size, channels);
            check_last_kernel("layernorm ln1");

            matmul(
                qkv_.get(), normalized_.get(),
                parameters_view_.qkvw +
                    static_cast<std::size_t>(layer) * 3 * channels * channels,
                parameters_view_.qkvb +
                    static_cast<std::size_t>(layer) * 3 * channels,
                batch_size, channels, 3 * channels);
            split_qkv_kernel<<<
                blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
                qkv_.get(), query_.get(), key_.get(), value_.get(),
                batch_size, channels);
            check_last_kernel("split_qkv_kernel");

            check_cuda(
                paged_attention_decode(
                    query_.get(), key_.get(), value_.get(),
                    key_cache_.get(), value_cache_.get(),
                    block_tables_.get(), context_lengths_.get(),
                    slot_mapping_.get(), attention_.get(), batch_size,
                    num_pages_, config_.num_layers, layer,
                    config_.num_heads, channels / config_.num_heads,
                    max_blocks_per_sequence_, max_context_length_,
                    stream_.get()),
                "paged_attention_decode");

            matmul(
                projected_.get(), attention_.get(),
                parameters_view_.attprojw +
                    static_cast<std::size_t>(layer) * channels * channels,
                parameters_view_.attprojb +
                    static_cast<std::size_t>(layer) * channels,
                batch_size, channels, channels);
            residual_kernel<<<
                blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
                residual_b_.get(), residual_a_.get(), projected_.get(),
                channel_elements);
            check_last_kernel("attention residual");

            layernorm_kernel<<<batch_size, kThreads, 0, stream_.get()>>>(
                normalized_.get(), residual_b_.get(),
                parameters_view_.ln2w +
                    static_cast<std::size_t>(layer) * channels,
                parameters_view_.ln2b +
                    static_cast<std::size_t>(layer) * channels,
                batch_size, channels);
            check_last_kernel("layernorm ln2");
            matmul(
                hidden_.get(), normalized_.get(),
                parameters_view_.fcw +
                    static_cast<std::size_t>(layer) *
                    hidden_width * channels,
                parameters_view_.fcb +
                    static_cast<std::size_t>(layer) * hidden_width,
                batch_size, channels, hidden_width);
            gelu_kernel<<<
                blocks_for(batch_size * hidden_width), kThreads, 0,
                stream_.get()>>>(
                hidden_.get(), batch_size * hidden_width);
            check_last_kernel("gelu_kernel");
            matmul(
                projected_.get(), hidden_.get(),
                parameters_view_.fcprojw +
                    static_cast<std::size_t>(layer) *
                    channels * hidden_width,
                parameters_view_.fcprojb +
                    static_cast<std::size_t>(layer) * channels,
                batch_size, hidden_width, channels);
            residual_kernel<<<
                blocks_for(channel_elements), kThreads, 0, stream_.get()>>>(
                residual_a_.get(), residual_b_.get(), projected_.get(),
                channel_elements);
            check_last_kernel("MLP residual");
        }

        layernorm_kernel<<<batch_size, kThreads, 0, stream_.get()>>>(
            normalized_.get(), residual_a_.get(), parameters_view_.lnfw,
            parameters_view_.lnfb, batch_size, channels);
        check_last_kernel("final layernorm");
        matmul(
            logits_.get(), normalized_.get(), parameters_view_.wte,
            nullptr, batch_size, channels, config_.padded_vocab_size);
        argmax_kernel<<<batch_size, kThreads, 0, stream_.get()>>>(
            logits_.get(), sampled_token_ids_.get(), batch_size,
            config_.vocab_size, config_.padded_vocab_size);
        check_last_kernel("argmax_kernel");

        std::vector<int> sampled(batch_size);
        check_cuda(
            cudaMemcpyAsync(
                sampled.data(), sampled_token_ids_.get(),
                sampled.size() * sizeof(int), cudaMemcpyDeviceToHost,
                stream_.get()),
            "copy sampled token ids");
        check_cuda(
            cudaStreamSynchronize(stream_.get()),
            "finish CUDA GPT-2 micro batch");
        last_batch_size_ = batch_size;
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
    DeviceBuffer<float> parameters_;
    ParameterViews parameters_view_;
    DeviceBuffer<int> token_ids_;
    DeviceBuffer<int> positions_;
    DeviceBuffer<int> context_lengths_;
    DeviceBuffer<int> slot_mapping_;
    DeviceBuffer<int> block_tables_;
    DeviceBuffer<int> sampled_token_ids_;
    DeviceBuffer<float> key_cache_;
    DeviceBuffer<float> value_cache_;
    DeviceBuffer<float> residual_a_;
    DeviceBuffer<float> residual_b_;
    DeviceBuffer<float> normalized_;
    DeviceBuffer<float> qkv_;
    DeviceBuffer<float> query_;
    DeviceBuffer<float> key_;
    DeviceBuffer<float> value_;
    DeviceBuffer<float> attention_;
    DeviceBuffer<float> projected_;
    DeviceBuffer<float> hidden_;
    DeviceBuffer<float> logits_;
    std::vector<ModelInput> last_model_inputs_;
    std::size_t last_host_to_device_bytes_ = 0;
    int last_batch_size_ = 0;
};

GPT2CudaModelRunner::GPT2CudaModelRunner(
    GPT2CudaConfig config, const float* host_parameters,
    std::size_t num_parameters, BlockManager& block_manager,
    std::size_t max_num_sequences,
    std::size_t max_num_batched_tokens,
    std::size_t max_context_length)
    : impl_(std::make_unique<Impl>(
          config, host_parameters, num_parameters, block_manager,
          max_num_sequences, max_num_batched_tokens,
          max_context_length)) {}

GPT2CudaModelRunner::~GPT2CudaModelRunner() = default;

std::vector<int> GPT2CudaModelRunner::run(const SchedulerOutput& output) {
    return impl_->run(output);
}

const std::vector<ModelInput>& GPT2CudaModelRunner::last_model_inputs() const {
    return impl_->last_model_inputs();
}

std::vector<float> GPT2CudaModelRunner::last_logits_for_testing() const {
    return impl_->last_logits_for_testing();
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

} // namespace cuda
} // namespace mini_vllm
