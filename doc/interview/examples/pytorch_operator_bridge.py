"""CPU/float64 算子语义对照；不加载 GPT-2 权重，不宣称替代 CUDA 数值回归。"""
import argparse
import math

import torch
from torch.nn import functional as F

PAGE, LAYERS, HEADS, DIM, CHANNELS = 16, 2, 2, 4, 8
LAYER, VOCAB, PADDED_VOCAB = 1, 13, 16


def fixture():
    generator = torch.Generator().manual_seed(20260911)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    # A 已计算17个、本轮1个；B 已计算15个、本轮4个。
    owners = torch.tensor([0, 1, 1, 1, 1])
    positions = torch.tensor([17, 15, 16, 17, 18])
    tables = torch.tensor([[5, 2], [3, 7]])
    row_tables = tables[owners]
    slots = row_tables.gather(1, (positions // PAGE)[:, None]).squeeze(1) * PAGE + positions % PAGE
    x, w, bias = rand(5, CHANNELS), rand(3 * CHANNELS, CHANNELS), rand(3 * CHANNELS)
    normalized = F.layer_norm(x, (CHANNELS,), eps=1e-5)
    projected = F.linear(normalized, w, bias)
    q, k, v = [part.reshape(5, HEADS, DIM).contiguous() for part in projected.chunk(3, -1)]
    # dense 的索引是 [请求,逻辑位置,head,dimension]；只预装已经计算的历史。
    dense_k, dense_v = rand(2, 19, HEADS, DIM), rand(2, 19, HEADS, DIM)
    cache_k = torch.full((8, LAYERS, HEADS, PAGE, DIM), -999.0, dtype=torch.float64)
    cache_v = torch.full_like(cache_k, -999.0)
    for owner, computed in enumerate((17, 15)):
        for position in range(computed):
            page, offset = tables[owner, position // PAGE].item(), position % PAGE
            cache_k[page, LAYER, :, offset, :] = dense_k[owner, position]
            cache_v[page, LAYER, :, offset, :] = dense_v[owner, position]
    return dict(owners=owners, positions=positions, tables=tables, row_tables=row_tables,
                slots=slots, x=x, w=w, bias=bias, normalized=normalized, projected=projected,
                q=q, k=k, v=v, dense_k=dense_k, dense_v=dense_v,
                cache_k=cache_k, cache_v=cache_v, wte=rand(PADDED_VOCAB, CHANNELS))


def write_pages(data):
    for row, slot in enumerate(data['slots'].tolist()):
        page, offset = divmod(slot, PAGE)
        data['cache_k'][page, LAYER, :, offset, :] = data['k'][row]
        data['cache_v'][page, LAYER, :, offset, :] = data['v'][row]


def write_dense(data):
    # 独立按请求与逻辑位置写，不复用页表或 slot，避免两条路径同时抄错映射。
    for row, (owner, position) in enumerate(zip(data['owners'].tolist(), data['positions'].tolist())):
        data['dense_k'][owner, position] = data['k'][row]
        data['dense_v'][owner, position] = data['v'][row]


def dense_attention(data):
    keys = data['dense_k'][data['owners']].permute(0, 2, 1, 3)  # [N,H,T,D]
    values = data['dense_v'][data['owners']].permute(0, 2, 1, 3)
    scores = torch.einsum('nhd,nhtd->nht', data['q'], keys) / math.sqrt(DIM)
    visible = torch.arange(keys.shape[2])[None, :] <= data['positions'][:, None]
    scores = scores.masked_fill(~visible[:, None, :], float('-inf'))
    return torch.einsum('nht,nhtd->nhd', scores.softmax(-1), values)


def paged_attention(data):
    output = []
    for row, position in enumerate(data['positions'].tolist()):
        logical = torch.arange(position + 1)
        pages = data['row_tables'][row, logical // PAGE]
        offsets = logical % PAGE
        # 高级索引输出 [context,H,D]，再变成 [H,context,D]。
        keys = data['cache_k'][pages, LAYER, :, offsets, :].transpose(0, 1)
        values = data['cache_v'][pages, LAYER, :, offsets, :].transpose(0, 1)
        scores = (data['q'][row, :, None, :] * keys).sum(-1) / math.sqrt(DIM)
        weights = (scores - scores.max(-1, keepdim=True).values).exp()
        weights = weights / weights.sum(-1, keepdim=True)
        output.append((weights[:, :, None] * values).sum(1))
    return torch.stack(output)


def qkv_lesson():
    data = fixture()
    # 对照 row-major Y=XW^T 与 cuBLAS 列主序视角 Y^T=WX^T。
    row_result = data['normalized'] @ data['w'].T
    column_result = data['w'] @ data['normalized'].T
    torch.testing.assert_close(row_result, column_result.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(row_result + data['bias'], data['projected'], atol=1e-12, rtol=1e-12)
    flat = data['projected'].flatten()
    # 用 CUDA split 的扁平地址单独检查 Q、K、V 三段。
    for row in range(5):
        for channel in range(CHANNELS):
            for part, name in enumerate(('q', 'k', 'v')):
                assert flat[row*3*CHANNELS + part*CHANNELS + channel] == data[name].reshape(5, -1)[row, channel]
    print('QKV PASS: X=(5,8), W=(24,8), projected=(5,24), Q/K/V=(5,2,4)')
    print('GEMM PASS: Y=XW^T 与 Y^T=WX^T；bias 独立广播')


def kv_lesson():
    data = fixture()
    before_k, before_v = data['cache_k'].clone(), data['cache_v'].clone()
    write_pages(data)
    assert data['slots'].tolist() == [33, 63, 112, 113, 114]
    touched = torch.zeros_like(data['cache_k'], dtype=torch.bool)
    for row, slot in enumerate(data['slots'].tolist()):
        page, offset = divmod(slot, PAGE)
        touched[page, LAYER, :, offset, :] = True
        torch.testing.assert_close(data['cache_k'][page, LAYER, :, offset, :], data['k'][row], atol=0, rtol=0)
        torch.testing.assert_close(data['cache_v'][page, LAYER, :, offset, :], data['v'][row], atol=0, rtol=0)
    assert torch.equal(before_k[~touched], data['cache_k'][~touched])
    assert torch.equal(before_v[~touched], data['cache_v'][~touched])
    page, layer, head, offset, dimension = 2, 1, 1, 1, 3
    flat_offset = ((((page*LAYERS + layer)*HEADS + head)*PAGE + offset)*DIM + dimension)
    assert flat_offset == 711
    assert data['cache_k'].flatten()[flat_offset] == data['k'][0, head, dimension]
    print('KV PASS: slots=[33,63,112,113,114]；只写目标层和槽，其他元素保持不变')
    print('地址 PASS: slot=33，但 K[page=2,layer=1,head=1,offset=1,d=3] 的元素偏移=711')


def attention_lesson():
    data = fixture()
    write_pages(data); write_dense(data)
    dense, paged = dense_attention(data), paged_attention(data)
    torch.testing.assert_close(paged, dense, atol=1e-12, rtol=1e-12)
    error = (dense-paged).abs().max().item()
    # 修改 B 位置16..18的未来V：不应影响 B 位置15，也不应影响 A。
    for position in (16, 17, 18):
        data['dense_v'][1, position] += 10000
        page = data['tables'][1, position // PAGE].item()
        data['cache_v'][page, LAYER, :, position % PAGE, :] += 10000
    changed = paged_attention(data)
    torch.testing.assert_close(changed[:2], paged[:2], atol=0, rtol=0)
    assert (changed[2:] - paged[2:]).abs().max() > 1
    torch.testing.assert_close(changed, dense_attention(data), atol=1e-10, rtol=1e-12)
    # 教学反例：把正确页表替换为“物理页从0连续排布”，必须被全量输出比较发现。
    data['row_tables'] = torch.tensor([[0, 1]]).repeat(5, 1)
    wrong = paged_attention(data)
    assert not torch.allclose(wrong, changed)
    print('Attention PASS: dense mask vs paged 逐行读取；最大误差=%.3g' % error)
    print('因果/隔离 PASS: 修改 B 的未来V，A与B较早行不变；错误页表被检测')


def head_lesson():
    data = fixture()
    write_pages(data)
    hidden = F.layer_norm(data['x'] + paged_attention(data).reshape(5, CHANNELS), (CHANNELS,), eps=1e-5)
    rows = torch.tensor([0, 4], dtype=torch.long)
    full = F.linear(hidden, data['wte'])
    gathered = hidden.index_select(0, rows)
    pruned = F.linear(gathered, data['wte'])
    torch.testing.assert_close(pruned, full.index_select(0, rows), atol=1e-12, rtol=1e-12)
    empty = F.linear(hidden[:0], data['wte'])
    assert empty.shape == (0, PADDED_VOCAB)
    samples = pruned[:, :VOCAB].argmax(-1)
    # padding 列数值不参与词表语义；即使很大也必须忽略。
    poisoned = pruned.clone(); poisoned[:, VOCAB:] = 1e9
    assert torch.equal(samples, poisoned[:, :VOCAB].argmax(-1))
    assert (poisoned.argmax(-1) >= VOCAB).all()
    # 只比较 Argmax 会漏掉这类数值差异。
    shifted = pruned + 100
    assert torch.equal(shifted[:, :VOCAB].argmax(-1), samples)
    assert not torch.allclose(shifted, pruned)
    print('LM Head PASS: (5,8)->(5,16) 对照 Gather [0,4]->(2,8)->(2,16)，R=0 输出(0,16)')
    print('采样 PASS: 忽略 padded 词表；相同 Argmax 仍可能有不同 logits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'qkv', 'kv', 'attention', 'head'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(1)
    lessons = {'qkv': qkv_lesson, 'kv': kv_lesson, 'attention': attention_lesson, 'head': head_lesson}
    with torch.no_grad():
        for name, lesson in lessons.items():
            if args.case in ('all', name):
                lesson()
    print('全部选定实验通过；本程序是 CPU 语义参考，不是 CUDA 精度或性能结论。')


if __name__ == '__main__':
    main()
