"""PyTorch 开发者的 CPU 入门实验；随机小模型，只验证原理，不作性能/文本质量基准。"""
import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


class TinyCausalLM(nn.Module):
    """双层、绝对位置、Pre-LN 的小型因果模型，省略 Dropout。"""

    def __init__(self):
        super().__init__()
        self.channels, self.heads = 16, 2
        self.wte = nn.Embedding(32, self.channels)
        self.wpe = nn.Embedding(64, self.channels)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(self.channels),
                "qkv": nn.Linear(self.channels, 3 * self.channels),
                "proj": nn.Linear(self.channels, self.channels),
                "ln2": nn.LayerNorm(self.channels),
                "mlp": nn.Sequential(nn.Linear(self.channels, 4 * self.channels),
                                     nn.GELU(), nn.Linear(4 * self.channels, self.channels)),
            }) for _ in range(2)
        ])
        self.lnf = nn.LayerNorm(self.channels)

    def forward(self, token_ids, past=None):
        # 输入是单请求本轮新增的 Token，形状 [Q]。past 每层含一对 [H,S,D] 的 K/V。
        old_length = 0 if past is None else past[0][0].shape[1]
        positions = torch.arange(old_length, old_length + token_ids.numel())
        x = self.wte(token_ids) + self.wpe(positions)
        updated = []
        for layer_id, layer in enumerate(self.layers):
            q, k, v = layer["qkv"](layer["ln1"](x)).chunk(3, dim=-1)
            q, k, v = [t.reshape(-1, self.heads, self.channels // self.heads).transpose(0, 1)
                       for t in (q, k, v)]
            if past is not None:
                k = torch.cat((past[layer_id][0], k), dim=1)
                v = torch.cat((past[layer_id][1], v), dim=1)
            # 关键：Q 的绝对位置从 old_length 开始。不能简单对矩形分数左上角 tril。
            key_positions = torch.arange(k.shape[1])
            visible = key_positions[None, :] <= positions[:, None]
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(~visible, float("-inf"))
            attended = scores.softmax(dim=-1) @ v
            attended = attended.transpose(0, 1).reshape(-1, self.channels)
            x = x + layer["proj"](attended)
            x = x + layer["mlp"](layer["ln2"](x))
            updated.append((k, v))
        logits = F.linear(self.lnf(x), self.wte.weight)
        return logits, updated


def generate(model, prompt, use_cache):
    tokens = prompt.clone()
    incoming = prompt
    cache = None
    processed_rows = 0
    generated = []
    for _ in range(3):
        inputs = incoming if use_cache else tokens
        logits, cache = model(inputs, cache if use_cache else None)
        processed_rows += inputs.numel()
        sampled = logits[-1].argmax().reshape(1)
        generated.append(sampled.item())
        tokens = torch.cat((tokens, sampled))
        incoming = sampled  # 刚采样的 Token 到下一轮才作为输入、产生自己的 KV。
    return generated, processed_rows


def cache_lesson():
    torch.manual_seed(7)
    model = TinyCausalLM().double().eval()
    tokens = torch.tensor([2, 5, 7, 11, 13, 17, 19, 23, 29])
    full_logits, full_cache = model(tokens)
    chunk_logits, cache, start = [], None, 0
    for size in (3, 2, 4):
        result, cache = model(tokens[start:start + size], cache)
        chunk_logits.append(result)
        start += size
        print("Chunk: 新输入=%d，已缓存=%d，每层 K 形状=%s" % (size, start, tuple(cache[0][0].shape)))
    actual = torch.cat(chunk_logits)
    torch.testing.assert_close(actual, full_logits, rtol=1e-10, atol=1e-10)
    for (ka, va), (kb, vb) in zip(cache, full_cache):
        torch.testing.assert_close(ka, kb, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(va, vb, rtol=1e-10, atol=1e-10)
    # 修改未来输入不影响更早位置的输出，直接展示因果性。
    altered = tokens.clone()
    altered[-1] = 3
    changed_logits, _ = model(altered)
    torch.testing.assert_close(changed_logits[:-1], full_logits[:-1], rtol=1e-10, atol=1e-10)
    no_cache, rows_a = generate(model, tokens[:5], False)
    with_cache, rows_b = generate(model, tokens[:5], True)
    assert no_cache == with_cache and (rows_a, rows_b) == (18, 7)
    print("完整前缀 vs 分块缓存: PASS，最大 logits 误差=%.3g" % (actual - full_logits).abs().max().item())
    print("修改未来 Token 不改变过去输出: PASS")
    print("Greedy 输出一致: %s；前向输入行数 %d → %d" % (with_cache, rows_a, rows_b))


def pages_lesson():
    torch.manual_seed(11)
    heads, length, dim, page_size = 2, 9, 8, 4
    k, v = [torch.randn(heads, length, dim, dtype=torch.float64) for _ in range(2)]
    q = torch.randn(heads, 1, dim, dtype=torch.float64)
    table = [3, 0, 4]  # 逻辑页依次分配到不连续物理页。
    k_pool, v_pool = [torch.full((5, heads, page_size, dim), float("nan"), dtype=torch.float64)
                      for _ in range(2)]
    for position in range(length):
        page, offset = table[position // page_size], position % page_size
        k_pool[page, :, offset] = k[:, position]
        v_pool[page, :, offset] = v[:, position]
    # 为展示数学等价性，Python 先 Gather 回连续张量；本项目 CUDA Kernel 直接按页访问。
    restored_k, restored_v = [torch.stack([
        pool[table[t // page_size], :, t % page_size] for t in range(length)
    ], dim=1) for pool in (k_pool, v_pool)]
    dense = (q @ k.transpose(-1, -2) / math.sqrt(dim)).softmax(-1) @ v
    paged = (q @ restored_k.transpose(-1, -2) / math.sqrt(dim)).softmax(-1) @ restored_v
    torch.testing.assert_close(restored_k, k, rtol=0, atol=0)
    torch.testing.assert_close(restored_v, v, rtol=0, atol=0)
    torch.testing.assert_close(paged, dense, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(paged).all()  # 未使用尾槽的 NaN 没有被读入。
    print("页表=%s，页大小=%d，Token 8 → 物理页 4 / 偏移 0 / Slot 16" % (table, page_size))
    print("连续 KV vs 分页 KV Attention: PASS，未读取页尾无效槽")


def packed_lesson():
    # A 已算 5 Token，本轮 Decode 1 个；B 新到达，本轮 Prefill 3 个。
    requests = [dict(tokens=[2, 5, 7, 11, 13, 17], computed=5, count=1, table=[3, 1]),
                dict(tokens=[19, 23, 29], computed=0, count=3, table=[4])]
    ids, positions, contexts, slots, query_starts, sample_rows = [], [], [], [], [0], []
    for request in requests:
        for p in range(request["computed"], request["computed"] + request["count"]):
            ids.append(request["tokens"][p])
            positions.append(p)
            contexts.append(p + 1)
            slots.append(request["table"][p // 4] * 4 + p % 4)
        query_starts.append(len(ids))
        if request["computed"] + request["count"] == len(request["tokens"]):
            sample_rows.append(len(ids) - 1)
    assert ids == [17, 19, 23, 29] and positions == [5, 0, 1, 2]
    assert contexts == [6, 1, 2, 3] and slots == [5, 16, 17, 18]
    assert query_starts == [0, 1, 4] and sample_rows == [0, 3]
    torch.manual_seed(13)
    hidden = torch.randn(4, 16, dtype=torch.float64)
    weight = torch.randn(32, 16, dtype=torch.float64)
    all_logits = F.linear(hidden, weight)
    chosen_logits = F.linear(hidden[sample_rows], weight)
    torch.testing.assert_close(chosen_logits, all_logits[sample_rows], rtol=1e-12, atol=1e-12)
    print("Packed Token=%s，Position=%s，Context=%s" % (ids, positions, contexts))
    print("Slot=%s，query_start=%s，sample_rows=%s" % (slots, query_starts, sample_rows))
    print("LM Head 裁剪前后所选行一致: PASS；请求数=2，输入行数=4，采样行数=2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", choices=("cache", "pages", "packed", "all"), nargs="?", default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        for name, lesson in (("cache", cache_lesson), ("pages", pages_lesson), ("packed", packed_lesson)):
            if args.lesson in (name, "all"):
                print("\n=== %s ===" % name)
                lesson()


if __name__ == "__main__":
    main()
