#!/usr/bin/env python3
"""只读仓库实测数据；输出面试数字与可选的真实区间图，不启动 CUDA。"""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def read_csv(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('CSV 没有数据: ' + str(path))
    return rows

def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)

def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('区间结束早于开始')
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result

def overlap_ns(left, right):
    a, b = merge(left), merge(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total

def plot_sample(samples, destination):
    # 可选依赖，仅画图时导入；统计分析只依赖标准库。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        raise RuntimeError("绘制中文图需要 Noto Sans CJK；可省略 --svg 先运行统计。")
    font = FontProperties(fname=str(font_path))
    first = samples[0]
    anchor = int(first['gpu1_start_ns'])
    chosen = [row for row in samples if int(row['gpu1_start_ns']) == anchor]
    fig, ax = plt.subplots(figsize=(10, 3))
    for gpu, color in [(0, '#2563eb'), (1, '#f59e0b')]:
        intervals = sorted({(int(row['gpu%d_start_ns' % gpu]),
                            int(row['gpu%d_end_ns' % gpu])) for row in chosen})
        ax.broken_barh([((s-anchor)/1000, (e-s)/1000) for s,e in intervals],
                       (gpu-0.18, 0.36), facecolors=color)
    start = max(int(first['gpu0_start_ns']), int(first['gpu1_start_ns']))
    end = min(int(first['gpu0_end_ns']), int(first['gpu1_end_ns']))
    ax.axvspan((start-anchor)/1000, (end-anchor)/1000, color='#16a34a', alpha=0.25)
    ax.annotate('首对区间重叠：%.3f 微秒' % ((end-start)/1000),
                xy=((start+end-2*anchor)/2000, 0.2), xytext=(4,0.5),
                arrowprops={'arrowstyle':'->'}, fontsize=10, fontproperties=font)
    ax.set_yticks([0,1]);ax.set_yticklabels(['GPU 0：样本中的算子', 'GPU 1：Argmax'], fontproperties=font)
    ax.set_xlabel('相对所选 GPU 1 算子起点的时间（微秒）', fontproperties=font)
    ax.set_title('真实 PD 时间线局部：仅展示所选样本区间', fontproperties=font)
    ax.grid(axis='x', alpha=0.25);fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(destination), metadata={'Date': None})
    plt.close(fig)
    if destination.suffix.lower() == '.svg':
        # 清除 Matplotlib 在路径坐标行末添加的空格，便于版本管理。
        svg = destination.read_text(encoding='utf-8')
        destination.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n',
                               encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'benchmark/results/task09_11')
    parser.add_argument('--svg', type=Path)
    args = parser.parse_args()
    root = args.results
    print('采样行 A/B：中位数基于各文件原始 total_ms')
    for suffix, name in [('', 'Eager'), ('_graph', 'Graph')]:
        full=read_csv(root/('rows_full'+suffix+'.csv'))
        pruned=read_csv(root/('rows_pruned'+suffix+'.csv'))
        a,b=median(full,'total_ms'),median(pruned,'total_ms')
        print('%s n=%d/%d full=%.6f ms pruned=%.6f ms reduction=%.3f%%' %
              (name,len(full),len(pruned),a,b,100*(a-b)/a))
        print('  metadata H2D median: %.0f -> %.0f bytes' %
              (median(full,'metadata_h2d_bytes'),median(pruned,'metadata_h2d_bytes')))
    prefix=read_csv(root/'prefix.csv')
    for length in sorted({int(row['prefix_tokens']) for row in prefix}):
        miss=[r for r in prefix if int(r['prefix_tokens'])==length and r['mode']=='miss']
        hit=[r for r in prefix if int(r['prefix_tokens'])==length and r['mode']=='hit']
        a,b=median(miss,'ttft_ms'),median(hit,'ttft_ms')
        print('prefix=%d TTFT miss=%.6f hit=%.6f ms reduction=%.3f%% inputs=%.0f->%.0f' %
              (length,a,b,100*(a-b)/a,median(miss,'scheduled_tokens'),median(hit,'scheduled_tokens')))
    for filename in ['pd.csv','pd_graph.csv']:
        rows=read_csv(root/filename)
        single=[r for r in rows if r['mode']=='single'];pd=[r for r in rows if r['mode']=='pd']
        a,b=median(single,'total_ms'),median(pd,'total_ms')
        print('%s single=%.6f pd=%.6f ms ratio=%.3fx transfer=%.6f ms payload=%.0f host_bytes=%.0f' %
              (filename,a,b,b/a,median(pd,'transfer_ms'),median(pd,'kv_payload_bytes'),median(pd,'host_transfer_bytes')))
    samples=read_csv(root/'nsys_overlap_samples.csv')
    for row in samples:
        expected=max(0,min(int(row['gpu0_end_ns']),int(row['gpu1_end_ns']))-
                       max(int(row['gpu0_start_ns']),int(row['gpu1_start_ns'])))
        if expected != int(row['overlap_ns']):
            raise ValueError('配对重叠长度与原始时间不一致')
    first=samples[0]
    print('首对 kernel 交集: %d ns = %.3f us' % (int(first['overlap_ns']),int(first['overlap_ns'])/1000))
    summary=json.loads((root/'nsys_overlap.json').read_text())
    print('已存整次 trace 重叠: %.6f ms；%d 条配对样本不能替代完整轨迹。' %
          (summary['simultaneous_kernel_busy_ms'],len(samples)))
    # 教学反例：两组区间内部重叠时，配对求和会重复计算。
    left=[(0,10),(5,15)];right=[(7,12)]
    naive=sum(max(0,min(b,d)-max(a,c)) for a,b in left for c,d in right)
    correct=overlap_ns(left,right)
    assert naive==8 and correct==5
    print('教学区间: 配对直接相加=%d，先合并再求交=%d（任意时间单位）' % (naive,correct))
    if args.svg:
        plot_sample(samples,args.svg)
        print('真实样本区间图: '+str(args.svg))

if __name__ == '__main__':
    main()
