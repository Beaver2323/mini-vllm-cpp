#ifndef PAGED_KV_CACHE_HPP
#define PAGED_KV_CACHE_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

// 定义每个物理页的大小（即一个 Page 可以存放多少个 Token 的 KV 向量）
#define PAGE_SIZE 16

// KVCachePool (KV缓存池): 负责预先分配和管理所有的物理内存页。
// 在大语言模型推理中，预分配可以极大地减少动态内存申请的开销。
struct KVCachePool {
    int num_pages;   // 缓存池中总物理页的数量
    int page_size;   // 每个页能存的 token 数 (PAGE_SIZE)
    int num_layers;  // 模型的层数
    int num_heads;   // 模型的注意力头数
    int head_size;   // 每个注意力头的维度大小 (C / num_heads)

    // 连续分配的底层物理内存块，类似于操作系统中的物理内存
    float* k_cache; // 维度形状: [num_pages, num_layers, num_heads, page_size, head_size]
    float* v_cache; // 维度形状: [num_pages, num_layers, num_heads, page_size, head_size]

    std::vector<int> free_pages; // 空闲物理页的索引栈
    int num_free_pages;          // 当前剩余空闲页数量

    KVCachePool(int num_pages, int num_layers, int num_heads, int head_size)
        : num_pages(num_pages), page_size(PAGE_SIZE), num_layers(num_layers),
          num_heads(num_heads), head_size(head_size) {

        // 计算所需分配的总浮点数并分配内存
        size_t cache_elements = (size_t)num_pages * num_layers * num_heads * PAGE_SIZE * head_size;
        k_cache = (float*)malloc(cache_elements * sizeof(float));
        v_cache = (float*)malloc(cache_elements * sizeof(float));
        if (!k_cache || !v_cache) {
            std::cerr << "Error: Memory allocation failed for KV cache.\n";
            exit(EXIT_FAILURE);
        }

        // 初始化空闲页栈，索引从 num_pages-1 到 0（后进先出）
        free_pages.resize(num_pages);
        for (int i = 0; i < num_pages; i++) {
            free_pages[i] = num_pages - 1 - i; // Stack behavior，ps：栈行为利用时间局部性
        }
        num_free_pages = num_pages;
    }

    ~KVCachePool() {
        free(k_cache);
        free(v_cache);
    }

    // 从池中申请一个新的物理页，返回该页的物理索引
    int allocate_page() {
        if (num_free_pages == 0) {
            std::cerr << "Error: Out of memory pages in KV cache pool.\n";
            exit(EXIT_FAILURE);
        }
        return free_pages[--num_free_pages];//ps：移动栈帧，栈帧初始化在高位
    }
};


// PageTable (页表): 维护从逻辑序列 (Sequence) 到物理页 (Physical Page) 的映射。
// 类似操作系统的虚拟内存页表，不同序列的 Token 可以存储在不连续的物理页中。
struct PageTable {
    std::vector<int> block_tables; // 映射表, 维度: [B, max_blocks_per_seq], 记录了每个 batch 各自的物理页索引
    int max_blocks_per_seq;        // 单个序列最大允许分配的块（页）数
    std::vector<int> context_lengths; // 当前每个序列已生成的上下文长度 (Token 数量), 维度: [B]

    PageTable(int B, int max_blocks_per_seq) : max_blocks_per_seq(max_blocks_per_seq) {
        block_tables.resize(B * max_blocks_per_seq, 0);
        context_lengths.resize(B, 0);
    }
};

// 实现了 Paged Attention 的前向传播算子。
// 它能够在物理内存不连续的情况下，正确计算 Query 与所有历史 Keys 的注意力得分，
// 并按权重对 Values 进行求和。
inline void paged_attention_forward(float* out,
                                    float* qkv,
                                    KVCachePool* pool, PageTable* page_table, float* att_buffer,
                                    int layer_idx, int B, int seq_len, int C, int NH, int max_seq_len) {
    int hs = C / NH;//将一个token的总特征维度，分给每一个注意力头
    float scale = 1.0f / sqrtf(hs);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            const int sequence_length = page_table->context_lengths.empty()
                                            ? seq_len
                                            : page_table->context_lengths[b];
            //ps：找当前头的起始Q的位置
            float* query = qkv + b * (3*C) + h * hs; // 当前最新 Token 的 Query 向量，qkv是一整个batch的qkv的起始指针，3*C是QKV各一个C
            // ！每连续 16 个逻辑 Token，存放在同一个物理内存页中
            // 步骤 1: 将当前新生成的 K 和 V 写入到 KV 缓存池中
            int current_pos = sequence_length - 1; // 当前正在处理的 token 在序列中的位置（逻辑位置）
            //逻辑页号->物理页号
            int page_idx = page_table->block_tables[b * page_table->max_blocks_per_seq + current_pos / pool->page_size]; // 查询页表找物理页号
            int page_offset = current_pos % pool->page_size; // 计算在当前物理页内的偏移量

            // 计算物理内存中的目标地址
            size_t cache_offset = (size_t)page_idx * pool->num_layers * pool->num_heads * pool->page_size * hs
                                + (size_t)layer_idx * pool->num_heads * pool->page_size * hs
                                + (size_t)h * pool->page_size * hs
                            + (size_t)page_offset * hs;

            float* k_cache_target = pool->k_cache + cache_offset;
            float* v_cache_target = pool->v_cache + cache_offset;

            float* key = qkv + b * (3*C) + h * hs + C;     // 当前的 Key
            float* value = qkv + b * (3*C) + h * hs + C*2; // 当前的 Value

            for (int i = 0; i < hs; i++) {
                k_cache_target[i] = key[i];
                v_cache_target[i] = value[i];
            }

            // 步骤 2: 计算当前 Query 与历史所有 Keys 的未归一化注意力得分 (Pre-attention)
            float maxval = -10000.0f; // 用于 Softmax 的数值稳定性
            float* preatt = att_buffer + b * NH * max_seq_len + h * max_seq_len; // 使用 att_buffer 保存当前头的计算结果

            // 遍历所有过去的 token（包括刚刚写入的最新 token）
            for (int t2 = 0; t2 < sequence_length; t2++) {
                int p_idx = page_table->block_tables[b * page_table->max_blocks_per_seq + t2 / pool->page_size]; // 查表获取历史 Token 的物理页
                int p_offset = t2 % pool->page_size; // 获取页内偏移
                size_t k_offset = (size_t)p_idx * pool->num_layers * pool->num_heads * pool->page_size * hs
                                + (size_t)layer_idx * pool->num_heads * pool->page_size * hs
                                + (size_t)h * pool->page_size * hs
                                + (size_t)p_offset * hs;
                float* k_t2 = pool->k_cache + k_offset;

                float val = 0.0f; // 计算 Query 和 K_t2 的点积
                for (int i = 0; i < hs; i++) {
                    val += query[i] * k_t2[i];
                }
                val *= scale;
                if (val > maxval) maxval = val;
                preatt[t2] = val;
            }

            // 步骤 3: 归一化注意力得分 (Softmax 的 exp 和 sum 阶段)
            float expsum = 0.0f;
            for (int t2 = 0; t2 < sequence_length; t2++) {
                float expv = expf(preatt[t2] - maxval);
                expsum += expv;
                preatt[t2] = expv; // store att directly in preatt buffer
            }
            float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

            // 步骤 4: 根据归一化后的注意力权重，累加历史所有的 Values (Normalize and accumulate)
            float* out_bth = out + b * C + h * hs;
            for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;

            for (int t2 = 0; t2 < sequence_length; t2++) {
                float a = preatt[t2] * expsum_inv; // 最终的 Attention 权重

                // 同样通过页表从不连续的物理内存中获取对应的 V 向量
                int p_idx = page_table->block_tables[b * page_table->max_blocks_per_seq + t2 / pool->page_size];
                int p_offset = t2 % pool->page_size;
                size_t v_offset = (size_t)p_idx * pool->num_layers * pool->num_heads * pool->page_size * hs
                                + (size_t)layer_idx * pool->num_heads * pool->page_size * hs
                                + (size_t)h * pool->page_size * hs
                                + (size_t)p_offset * hs;
                float* v_t2 = pool->v_cache + v_offset;
                for (int i = 0; i < hs; i++) {
                    out_bth[i] += a * v_t2[i];
                }
            }
        }
    }
}

#endif // PAGED_KV_CACHE_HPP
