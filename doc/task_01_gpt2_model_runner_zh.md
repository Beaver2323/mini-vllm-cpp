# 开发任务 01：接通 Scheduler 与 GPT2ModelRunner

**状态：已完成。**

## 为什么这是下一步

项目现在已经有两部分能力：

1. Sequence、BlockManager、Scheduler 能决定“本轮计算哪些请求、计算多少 Token”。
2. GPT-2 增量前向能根据每个请求独立的 context length 读写分页 KV Cache。

本任务已经通过统一 ModelRunner 接口连接这两部分：调度结果会转换为模型输入，模型输出
经过 greedy sampling 后提交回 Scheduler，完成请求状态更新和 Block 回收。

## 需要建立的数据结构

```cpp
struct ModelInput {
    std::vector<int> token_ids;
    std::vector<int> positions;
    std::vector<int> context_lengths;
    std::vector<int> slot_mapping;
    std::vector<int> block_tables;
    std::vector<std::uint64_t> request_ids;
};
```

字段含义：

- `token_ids`：本轮真正送入模型的 Token。
- `positions`：Token 在所属请求中的绝对位置。
- `context_lengths`：各请求本轮可见的上下文长度。
- `slot_mapping`：新 K/V 写入的物理页号和页内偏移。
- `block_tables`：逻辑页号到物理页号的映射。
- `request_ids`：模型输出与 Sequence 之间的对应关系。

## 实现步骤

### 1. 独立推理工作区

新增只服务于增量推理的 Workspace，至少保存：

- 单 Token 的 residual、LayerNorm、QKV、Attention 和 MLP 中间结果。
- 大小为 `max_active_sequences × num_heads × max_context_length` 的 Attention scratch。
- 大小为 `max_active_sequences × padded_vocab_size` 的 logits。

这样 ModelRunner 不需要先调用训练前向来间接分配激活内存。

### 2. 构造模型输入

ModelRunner 遍历 `SchedulerOutput.items`：

1. 从 Sequence 的 `num_computed_tokens` 找到本轮起始位置。
2. 读取对应的 Token。
3. 计算绝对 position 和 context length。
4. 从 Sequence 的 Block Table 计算物理页及页内 offset。
5. 将不同请求压紧成当前活跃 Batch。

如果某个 Prefill chunk 包含多个 Token，第一版可在 ModelRunner 内按 position 拆成多个
微步。正确性稳定后，再实现一次处理多个 Prefill Token 的专用路径。

### 3. 对齐 BlockManager 与 KVCachePool

BlockManager 分配的 Block ID 必须直接索引 KVCachePool 的物理页。ModelRunner 不再调用
KVCachePool 自己的页分配接口，避免出现两套互不一致的页所有权。

### 4. 执行和采样

每个微步调用 GPT-2 增量前向。只有当 ScheduledItem 已经计算完当前全部输入 Token 时，
才从最后一个位置的 logits 执行 greedy argmax，并返回 sampled token；部分 Prefill
返回 `-1`。

### 5. Engine 循环

```cpp
while (!scheduler.is_finished()) {
    SchedulerOutput output = scheduler.schedule();
    std::vector<int> sampled = model_runner.run(output);
    scheduler.commit(output, sampled);
}
```

这个循环是学习 vLLM 主链路的最小闭环。

## 验收标准

- [x] 两个以上异长请求能够在不同时间加入。
- [x] 同一轮能够同时处理至少一个 Decode 请求和一个 Chunked Prefill 请求。
- [x] 请求完成后 Block 数量立即恢复，后续请求能够复用这些 Block。
- [x] 每个请求生成的 greedy token 与独立完整前缀重算结果一致。
- [x] 覆盖第 16→17 Token 的跨页扩容；既有模型测试覆盖第 32→33 Token。
- [x] Scheduler 无法产生工作时由 Engine 报错，避免静默死循环。
- [x] AddressSanitizer 和 UndefinedBehaviorSanitizer 检查通过。

端到端测试入口为 `dev/test_gpt2_engine.cpp`。测试还验证了独立推理 Workspace：
该场景使用 435,556 个模型激活元素，而 B=3、T=18 的完整前缀 reference 使用
14,756,418 个激活元素。这个数字只描述该固定测试形状，不是通用显存节省比例。

## 完成后再做什么

1. 建立可重复 Benchmark，报告 TTFT、TPOT、总吞吐、P50/P95 和 KV Cache 使用量。
2. 将已完成的独立 CUDA Decode PagedAttention 接入 GPU ModelRunner。
3. 实现 Prefix Cache、Block 引用计数和抢占。
4. 在主链路稳定后再加入 Tensor Parallel 与 CUDA Graph。
