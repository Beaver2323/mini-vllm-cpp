# 第 1 节：从 model.forward 到自回归生成与 KV Cache

[返回学习入口](README.md) · 下一节：[请求与调度](02_requests_and_scheduler.md)

你只需要会读 `nn.Module`、矩阵乘法、reshape 和 Softmax。本节先处理一个请求，暂不讨论
多个用户争用 GPU。

## 1. 先明确模型究竟输出了什么

语言模型把 Token ID 转成 hidden state，再经过词表投影得到 logits。以单个长度 5 的输入为例：

```text
Token IDs         [1, 5]       # B=1，T=5
Embedding         [1, 5, C]
Transformer       [1, 5, C]
LM Head logits    [1, 5, V]
```

`logits[0, i]` 是“看到位置 0..i 后，下一个 Token 的词表分数”。它不是当前 Token 的 ID。
要接着整个 Prompt 生成，先使用最后一个输入位置的 logits。

下面是说明接口含义的伪代码；真实模型返回值可能是 Tensor、tuple 或带 logits 字段的对象：

```python
model.eval()
with torch.inference_mode():
    logits = model(prompt_ids)          # 假设返回 [B,T,V]
    next_id = logits[:, -1].argmax(-1)   # [B]，Greedy
```

`eval()` 改变 Dropout 等模块行为；`inference_mode()` 关闭相关 Autograd 跟踪。两者都不会自动
创建 KV Cache，也不会替你处理多请求调度。训练时用所有位置的 logits 算 loss；普通生成只需
当前有效输入末尾的位置，任务 09 的采样行裁剪正利用这一点。

## 2. 为什么生成需要循环

训练时，答案 Token 已经在样本里，可把已知序列右移后并行预测各位置。生成时，下一个 Token
尚未确定，要先从当前位置的 logits 选择它，再把它作为下一轮输入的一部分。

最直观、每次重算整个前缀的写法：

```python
# 教学伪代码：不包含 KV 参数，model 返回 Tensor。
tokens = prompt_ids
for _ in range(max_new_tokens):
    logits = model(tokens)
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    tokens = torch.cat([tokens, next_id], dim=1)
```

Prompt 长 5，生成 3 个新 Token 时，三次前向分别处理长度 5、6、7，共处理 18 个输入位置。
第三次前向生成最后一个输出后就停止，因此最后一个输出 Token 不需要再执行前向。

这里“18 行”统计的是每层需处理的输入 Token 行数，并非 Attention FLOPs，也不是 GPU Kernel
数量。后面不要把行数减少直接换算成相同倍数的加速。

## 3. Prefill 与 Decode 从哪里出现

Prefill（预填充）：处理 Prompt 的输入位置，建立历史状态，并在 Prompt 全部算完时产生首个
输出 Token。Decode（逐步解码）：用已有历史状态处理后续新输入，继续生成。

```text
Prompt: [p0,p1,p2,p3,p4]

Prefill 输入 p0..p4 → 得到 y0
Decode  输入 y0    → 得到 y1
Decode  输入 y1    → 得到 y2 → 达到输出上限，结束
```

Prefill 也可以拆成多个 Chunk。只完成一部分 Prompt 时还不能按这套生成流程输出 y0。
Prefill/Decode 描述的是本轮工作，不要求必须有两份模型，也不要求两张 GPU。单卡引擎本来就
可以顺序执行这两个阶段。

## 4. 从你熟悉的 Attention 公式推导缓存

先看一层、一个请求，省略 batch 维。投影和按头拆分后：

```text
Q: [H,Q_len,D]
K: [H,K_len,D]
V: [H,K_len,D]
Attention(Q,K,V) = softmax(QKᵀ / sqrt(D) + mask) V
```

无历史缓存的完整 Prefill 中，`Q_len = K_len = Prompt 长度`。有 5 个历史 Token 时，下一步
Decode 的 `Q_len=1`，但 `K_len=6`：新位置要查询旧的 5 个 Token 和自己。

因果 Attention 的位置 i 看不到未来。固定权重、位置编码、mask 和确定性推理行为时，追加
Token 不改变旧位置已计算出的 hidden state，也不改变各层投影出的旧 K/V。因此可以缓存旧
K/V，只投影新输入：

```python
# 下面是完整 CPU 实验中的实际语句。
q, k, v = layer["qkv"](layer["ln1"](x)).chunk(3, dim=-1)
# reshape 为 [H,Q_len,D] 后：
if past is not None:
    k = torch.cat((past[layer_id][0], k), dim=1)
    v = torch.cat((past[layer_id][1], v), dim=1)
```

历史 Q 不缓存，是因为本轮只需要新位置的 Q，查询旧 K/V 后得到新位置输出。不是说 Q 的数值
没有复用可能，而是这个自回归计算路径不再需要历史位置的 Attention 输出。

这份教学脚本用 `torch.cat` 便于看懂，真实引擎使用预分配缓存和原位写入，避免每轮搬迁整个
历史。下一节的状态管理与第 3 节的分页池会替换这个“不断变长的 Tensor”实现。

## 5. KV 必须每一层都保存

第 0 层的 K/V 来自第 0 层输入，第 1 层的 K/V 来自经过第 0 层后的 hidden state，它们不是
同一组数值。完整推理缓存一般按层组织：

```python
past = [
    (k_layer0, v_layer0),   # 每个都是 [H,T,D]
    (k_layer1, v_layer1),
]
```

本项目 GPT-2 有 12 层，每层 12 头、每头 64 维。FP16 下一个 Token 的总缓存大小为：

```text
2（K 与 V）× 12（层）× 12（KV 头）× 64（每头维度）× 2（字节）= 36 KiB
```

这是 KV 数据本身，不含权重、激活、页表、图池和显存分配器保留空间。以后遇到 GQA/MQA 时，
公式中的头数要用 **KV 头数**，不能直接使用 Q 的头数。本节的小模型和项目 GPT-2 都是 MHA。

## 6. 最容易漏掉的两个索引问题

**位置编码必须连续。** 5 Token Prompt 后输入 y0，位置编号是 5，不能因为当前输入长度为 1
就重新用位置 0。本项目用绝对位置 Embedding，实际生产位置的代码是：

```cpp
const std::size_t position = sequence.num_computed_tokens() + offset;
```

位置生产者在 [prepare_packed_model_input](../../mini_vllm/model_input.hpp#L41)，消费者是 CUDA
`embedding_kernel`。以后改用 RoPE，位置语义仍然重要，但编码算子会不同。

**带历史缓存的矩形 mask 不能从左上角重新开始。** 已有 3 个历史 Token，本轮输入两个新
Token，Attention 分数是 `[2,5]`，正确可见性为：

```text
          K0 K1 K2 K3 K4
新 Q3     1  1  1  1  0
新 Q4     1  1  1  1  1
```

实际实验用绝对位置构造可见性，避免不同 Attention API 对非方形 causal mask 的约定差异：

```python
positions = torch.arange(old_length, old_length + token_ids.numel())
key_positions = torch.arange(k.shape[1])
visible = key_positions[None, :] <= positions[:, None]
scores = scores.masked_fill(~visible, float("-inf"))
```

项目 CUDA 则把每行可见长度记为 `position+1`，Kernel 只遍历该长度内的历史位置。两种写法
表达同一个因果约束。

## 7. 亲自跑一个普通 PyTorch 模型

完整可运行文件：[attention_and_pages.py](examples/attention_and_pages.py)。定位顺序：

| 学习点 | 函数 | 调用点 |
| --- | --- | --- |
| 构造普通 Module | `TinyCausalLM.__init__` | `cache_lesson` |
| 前向、分头、每层缓存 | `TinyCausalLM.forward` | `cache_lesson` 与 `generate` |
| 无缓存/有缓存生成循环 | `generate` | `cache_lesson` 最后两次调用 |
| 验证分块一致性和因果性 | `cache_lesson` | CLI 的 `cache` 分支 |

```bash
conda activate zyf1
python doc/from_pytorch/examples/attention_and_pages.py cache
```

观察这些输出：

```text
Chunk: 新输入=3，已缓存=3，每层 K 形状=(2, 3, 8)
Chunk: 新输入=2，已缓存=5，每层 K 形状=(2, 5, 8)
Chunk: 新输入=4，已缓存=9，每层 K 形状=(2, 9, 8)
完整前缀 vs 分块缓存: PASS
修改未来 Token 不改变过去输出: PASS
Greedy 输出一致: [13, 13, 9]；前向输入行数 18 → 7
```

固定种子的小模型输出只是随机模型的 Token ID。学习目标是检查完整前缀与分块路径的整行
logits、各层 KV 和 Greedy 输出一致，而不是生成自然语言。

## 8. 把实验对应到项目代码

| PyTorch 小实验 | 项目实现位置 | 入口怎样到达这里 |
| --- | --- | --- |
| `generate` 循环 | [GPT2CudaEngine::step](../../mini_vllm/cuda/gpt2_cuda_engine.hpp#L75) | 外部 while 调用 step |
| 保存 token_ids 和长度 | [Sequence](../../mini_vllm/sequence.hpp#L21) | Engine::add_request 构造 |
| 本轮输入位置 | [prepare_packed_model_input](../../mini_vllm/model_input.hpp#L41) | Runner::run 调用 |
| LN、QKV、MLP | [Runner::forward](../../mini_vllm/cuda/gpt2_cuda_model_runner.cu)，搜索 `template <typename T>` / `forward` | Impl::run 按 dtype 分发 |
| `torch.cat` 旧新 KV | [write_kv_cache_kernel](../../mini_vllm/cuda/paged_attention.cu#L57) 的原位写入替代它 | paged_attention_decode_impl 先 Launch 写入 |
| `softmax(QKᵀ)V` | 同文件 `paged_attention_kernel` | 写入 Kernel 后在同 Stream Launch |
| next_id 追加 | [Scheduler::commit](../../mini_vllm/scheduler.hpp#L87) | Runner 返回后调用 |

## 9. 过关问题

先自己回答，再看第 6 节答案：

1. Prompt 长 5，生成 3 个 Token，无缓存为什么处理 18 行，有缓存为什么是 7 行？
2. y0 已经生成，为什么仍然没有它的 KV？
3. 同一请求中旧 K/V 可以复用，任意另一个请求中的相同 Token 就一定能复用吗？
4. 已有 3 Token 缓存，本轮处理 2 Token，Q/K/V 与分数张量各是什么形状？
5. 为什么 `model.eval()` 不等于“启用 KV Cache”？

你能画出 `[p0..p4] → y0 → y1 → y2` 的输入与 KV 生命周期，就可以进入下一节。
