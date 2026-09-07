#ifndef MINI_VLLM_TRAIN_GPT2_CPP
#define MINI_VLLM_TRAIN_GPT2_CPP

// Enable TESTING macro to prevent compiling the int main() from the included C file.
// This gives us access to all the original structs, weights, and layers.
#define TESTING
#include "train_gpt2.c"
#include "paged_kv_cache.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

// Re-implement sampling utilities which are skipped by #define TESTING
unsigned int random_u32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

// 增量推理专用工作区。它按 T=1 分配模型中间结果，并单独分配线性大小的
// Attention scratch，避免为了推理而创建训练前向所需的 T² 激活内存。
class GPT2InferenceWorkspace {
public:
    GPT2InferenceWorkspace(GPT2Config config, int max_batch_size,
                           int max_context_length)
        : max_batch_size_(max_batch_size),
          max_context_length_(max_context_length) {
        if (max_batch_size <= 0 || max_context_length <= 0 ||
            max_context_length > config.max_seq_len) {
            throw std::invalid_argument("invalid GPT-2 inference workspace shape");
        }
        fill_in_activation_sizes(act_sizes_, config, max_batch_size, 1);
        // PagedAttention 使用独立 scratch；greedy decoding 直接对 logits
        // 取 argmax，不需要训练 Attention、概率和 loss 缓冲。
        act_sizes_[6] = 0;   // preatt
        act_sizes_[7] = 0;   // att
        act_sizes_[21] = 0;  // probs
        act_sizes_[22] = 0;  // losses
        for (std::size_t size : act_sizes_) {
            num_activations_ += size;
        }
        attention_scratch_.resize(
            static_cast<std::size_t>(max_batch_size) * config.num_heads *
            max_context_length);
        acts_memory_ = malloc_and_point_activations(&acts_, act_sizes_);
    }

    ~GPT2InferenceWorkspace() { free(acts_memory_); }

    GPT2InferenceWorkspace(const GPT2InferenceWorkspace&) = delete;
    GPT2InferenceWorkspace& operator=(const GPT2InferenceWorkspace&) = delete;

    int max_batch_size() const { return max_batch_size_; }
    int max_context_length() const { return max_context_length_; }
    std::size_t num_activations() const { return num_activations_; }
    ActivationTensors& acts() { return acts_; }
    const ActivationTensors& acts() const { return acts_; }
    float* attention_scratch() { return attention_scratch_.data(); }
    std::size_t attention_scratch_size() const {
        return attention_scratch_.size();
    }

private:
    int max_batch_size_;
    int max_context_length_;
    ActivationTensors acts_{};
    std::size_t act_sizes_[NUM_ACTIVATION_TENSORS]{};
    float* acts_memory_ = nullptr;
    std::size_t num_activations_ = 0;
    std::vector<float> attention_scratch_;
};

static void validate_paged_inference_inputs(
    const GPT2* model, const int* inputs, const KVCachePool* pool,
    const PageTable* page_table, int active_batch_size,
    int max_batch_size, int max_context_length,
    std::size_t attention_scratch_size) {
    if (model == nullptr || model->params_memory == nullptr || inputs == nullptr ||
        pool == nullptr || page_table == nullptr) {
        throw std::invalid_argument("paged inference received a null input");
    }
    if (active_batch_size <= 0 || active_batch_size > max_batch_size) {
        throw std::out_of_range("active inference batch exceeds workspace capacity");
    }
    if (max_context_length <= 0 ||
        max_context_length > model->config.max_seq_len) {
        throw std::out_of_range("inference context capacity exceeds model limit");
    }
    if (pool->num_layers != model->config.num_layers ||
        pool->num_heads != model->config.num_heads ||
        pool->head_size * pool->num_heads != model->config.channels) {
        throw std::invalid_argument("KV cache shape does not match GPT-2");
    }
    if (page_table->max_blocks_per_seq <= 0 ||
        page_table->context_lengths.size() <
            static_cast<std::size_t>(active_batch_size) ||
        page_table->block_tables.size() <
            static_cast<std::size_t>(active_batch_size) *
                page_table->max_blocks_per_seq) {
        throw std::invalid_argument("page table shape is invalid");
    }
    const std::size_t required_scratch =
        static_cast<std::size_t>(active_batch_size) *
        model->config.num_heads * max_context_length;
    if (attention_scratch_size < required_scratch) {
        throw std::out_of_range("attention scratch is too small");
    }

    for (int b = 0; b < active_batch_size; ++b) {
        if (inputs[b] < 0 || inputs[b] >= model->config.vocab_size) {
            throw std::out_of_range("input token is outside the vocabulary");
        }
        const int context_length = page_table->context_lengths[b];
        if (context_length <= 0 || context_length > max_context_length) {
            throw std::out_of_range("request context length is invalid");
        }
        const int required_blocks =
            (context_length + pool->page_size - 1) / pool->page_size;
        if (required_blocks > page_table->max_blocks_per_seq) {
            throw std::out_of_range("request exceeds page table capacity");
        }
        for (int logical_block = 0; logical_block < required_blocks;
             ++logical_block) {
            const int block_id =
                page_table->block_tables[
                    b * page_table->max_blocks_per_seq + logical_block];
            if (block_id < 0 || block_id >= pool->num_pages) {
                throw std::out_of_range("page table references an invalid block");
            }
        }
    }
}

// 专为自回归生成设计的单步前向传播。每个 Batch 行只包含一个新 Token，
// 而历史 K/V 通过各请求自己的 Block Table 和 context length 读取。
static void gpt2_forward_inference_impl(
    const GPT2* model, const int* inputs, KVCachePool* pool,
    const PageTable* page_table, int active_batch_size,
    ActivationTensors acts, float* attention_scratch,
    std::size_t attention_scratch_size, int layer_batch_capacity,
    int max_context_length, bool compute_probabilities) {
    validate_paged_inference_inputs(
        model, inputs, pool, page_table, active_batch_size,
        layer_batch_capacity, max_context_length, attention_scratch_size);

    size_t B = static_cast<size_t>(active_batch_size);
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    ParameterTensors params = model->params;
    const size_t layer_stride = static_cast<size_t>(layer_batch_capacity);

    float* encoded = acts.encoded;
    for (size_t b = 0; b < B; b++) {
        int ix = inputs[b]; // 提取各个序列当前刚生成的最新 token
        float* wte_ix = params.wte + ix * C;
        // T=1 only describes this invocation. Each request still uses its own
        // absolute position in the full sequence.
        int position = page_table->context_lengths[b] - 1;
        float* wpe_t = params.wpe + position * C;
        for (size_t i = 0; i < C; i++) {
            encoded[b * C + i] = wte_ix[i] + wpe_t[i];
        }
    }

    for (size_t l = 0; l < L; l++) {
        const size_t layer_offset = l * layer_stride;
        float* residual =
            l == 0 ? acts.encoded
                   : acts.residual3 + (l - 1) * layer_stride * C;

        float* l_ln1 = acts.ln1 + layer_offset * C;
        float* l_ln1_mean = acts.ln1_mean + layer_offset;
        float* l_ln1_rstd = acts.ln1_rstd + layer_offset;
        float* l_qkv = acts.qkv + layer_offset * 3 * C;
        float* l_atty = acts.atty + layer_offset * C;
        float* l_attproj = acts.attproj + layer_offset * C;
        float* l_residual2 = acts.residual2 + layer_offset * C;
        float* l_ln2 = acts.ln2 + layer_offset * C;
        float* l_ln2_mean = acts.ln2_mean + layer_offset;
        float* l_ln2_rstd = acts.ln2_rstd + layer_offset;
        float* l_fch = acts.fch + layer_offset * 4 * C;
        float* l_fch_gelu = acts.fch_gelu + layer_offset * 4 * C;
        float* l_fcproj = acts.fcproj + layer_offset * C;
        float* l_residual3 = acts.residual3 + layer_offset * C;

        // 逐层前向传播，但注意此时张量的序列长度 T=1
        // 因此我们在原版算子的最后一个参数传入了 1
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, params.ln1w + l*C, params.ln1b + l*C, B, 1, C);
        matmul_forward(l_qkv, l_ln1, params.qkvw + l*3*C*C, params.qkvb + l*3*C, B, 1, C, 3*C);

        // 使用我们新编写的 Paged Attention 替换掉原本原生的 attention_forward
        paged_attention_forward(
            l_atty, l_qkv, pool, page_table, attention_scratch,
            static_cast<int>(l), static_cast<int>(B), 0,
            static_cast<int>(C), static_cast<int>(NH), max_context_length);

        matmul_forward(l_attproj, l_atty, params.attprojw + l*C*C, params.attprojb + l*C, B, 1, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*C);

        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, params.ln2w + l*C, params.ln2b + l*C, B, 1, C);
        matmul_forward(l_fch, l_ln2, params.fcw + l*4*C*C, params.fcb + l*4*C, B, 1, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, params.fcprojw + l*C*4*C, params.fcprojb + l*C, B, 1, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*C);
    }

    layernorm_forward(
        acts.lnf, acts.lnf_mean, acts.lnf_rstd,
        acts.residual3 + (L - 1) * layer_stride * C,
        params.lnfw, params.lnfb, B, 1, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, 1, C, Vp);
    if (compute_probabilities) {
        softmax_forward(acts.probs, acts.logits, B, 1, V, Vp);
    }
}

void gpt2_forward_inference_with_workspace(
    const GPT2* model, const int* inputs, KVCachePool* pool,
    const PageTable* page_table, int active_batch_size,
    GPT2InferenceWorkspace* workspace) {
    if (workspace == nullptr) {
        throw std::invalid_argument("GPT-2 inference workspace is null");
    }
    gpt2_forward_inference_impl(
        model, inputs, pool, page_table, active_batch_size, workspace->acts(),
        workspace->attention_scratch(), workspace->attention_scratch_size(),
        workspace->max_batch_size(), workspace->max_context_length(),
        /*compute_probabilities=*/false);
}

// 兼容早期测试和示例：继续使用 model->acts，但新引擎应使用独立 Workspace。
void gpt2_forward_inference_batched(
    GPT2* model, int* inputs, KVCachePool* pool,
    PageTable* page_table, int active_batch_size) {
    if (model == nullptr || model->acts_memory == nullptr) {
        throw std::logic_error("model activations are not initialized");
    }
    gpt2_forward_inference_impl(
        model, inputs, pool, page_table, active_batch_size, model->acts,
        model->acts.preatt, model->act_sizes[6], active_batch_size,
        model->config.max_seq_len, /*compute_probabilities=*/true);
}

void gpt2_forward_inference(GPT2 *model, int* inputs, KVCachePool* pool,
                            PageTable* page_table, int seq_len) {
    (void)seq_len; // Kept for compatibility with the original single-length caller.
    gpt2_forward_inference_batched(model, inputs, pool, page_table,
                                   model->batch_size);
}

#ifndef GPT2_PAGED_INFERENCE_NO_MAIN
int main() {
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4;
    int T = 64;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    uint64_t rng_state = 1337;
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // 基于 Paged Attention 和 KV Cache 池的生成循环 (推理阶段)
        if (step > 0 && step % 20 == 0) {
            int page_size = PAGE_SIZE;
            // 计算生成目标长度最多需要用到多少个页（块）
            int max_blocks_per_seq = (genT + page_size - 1) / page_size;
            int total_pages = B * max_blocks_per_seq; // 为了简化演示，这里分配充足的总页数

            // 实例化 KV 缓存池以及关联的虚拟页表
            KVCachePool kv_pool(total_pages, model.config.num_layers, model.config.num_heads, model.config.channels / model.config.num_heads);
            PageTable page_table(B, max_blocks_per_seq);
            std::vector<int> current_tokens(B, tokenizer.eot_token);

            printf("generating:\n---\n");
            for (int t = 1; t <= genT; t++) {
                for (int b = 0; b < B; b++) {
                    int seq_len = page_table.context_lengths[b];
                    // Paged Attention 的核心特点：延迟按需分配（Lazy Allocation）
                    // 当序列长度刚好填满当前物理页时（或者是全新序列长度为 0 时），向内存池动态请求一个新的物理页。
                    if (seq_len % kv_pool.page_size == 0) {
                        page_table.block_tables[b * max_blocks_per_seq + seq_len / kv_pool.page_size] = kv_pool.allocate_page();
                    }
                    page_table.context_lengths[b]++;
                }

                // 使用携带 Paged Attention 机制的快速单步推理代替原本粗暴的重新计算
                gpt2_forward_inference(&model, current_tokens.data(), &kv_pool, &page_table, t);

                float* probs = model.acts.probs;
                int next_token = sample_mult(probs, model.config.vocab_size, random_f32(&rng_state));
                for (int b = 0; b < B; b++) current_tokens[b] = sample_mult(probs + b * model.config.padded_vocab_size, model.config.vocab_size, random_f32(&rng_state));
                current_tokens[0] = next_token; // 保证用于终端打印流的随机分支具有确定性

                if (t < genT) {
                    if (tokenizer.init_ok) safe_printf(tokenizer_decode(&tokenizer, next_token));
                    else printf("%d ", next_token);
                    fflush(stdout);
                }
            }
            printf("\n---\n");
        }

        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }
}
#endif

#endif // MINI_VLLM_TRAIN_GPT2_CPP
