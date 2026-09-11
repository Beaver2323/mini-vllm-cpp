#!/usr/bin/env python3
"""汇总 Nsight SQLite 中两张 GPU 的 Kernel 实际重叠，保留配对样本供核查。"""
import csv
import hashlib
import json
import pathlib
import sqlite3
import sys

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
connection = sqlite3.connect('file:' + str(source.resolve()) + '?mode=ro', uri=True)
rows = connection.execute('''
    SELECT k.start, k.end, k.deviceId, k.streamId, s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
    ORDER BY k.start
''').fetchall()
by_device = {device: [r for r in rows if r[2] == device] for device in (0, 1)}

def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result

unions = {device: merge((r[0], r[1]) for r in values) for device, values in by_device.items()}
a, b = unions[0], unions[1]
i = j = 0
overlap = 0
while i < len(a) and j < len(b):
    overlap += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
    if a[i][1] <= b[j][1]: i += 1
    else: j += 1

samples = []
j = 0
for left in by_device[0]:
    while j < len(by_device[1]) and by_device[1][j][1] <= left[0]: j += 1
    k = j
    while k < len(by_device[1]) and by_device[1][k][0] < left[1]:
        right = by_device[1][k]
        duration = min(left[1], right[1]) - max(left[0], right[0])
        if duration > 0:
            samples.append((left[0], left[1], right[0], right[1], duration, left[3], right[3], left[4], right[4]))
        k += 1
    if len(samples) >= 20: break
assert overlap > 0 and samples, '没有检测到双卡 Kernel 时间重叠'
with (output / 'nsys_overlap_samples.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['gpu0_start_ns', 'gpu0_end_ns', 'gpu1_start_ns', 'gpu1_end_ns', 'overlap_ns',
                     'gpu0_stream', 'gpu1_stream', 'gpu0_kernel', 'gpu1_kernel'])
    writer.writerows(samples[:20])
summary = {
    '说明': '同一次独立 profiler 运行，包含初始化、预热与七次测量；只证明真实 Kernel 重叠，不用 profiler 耗时宣称加速。',
    'sqlite_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'kernel_count': {str(d): len(r) for d, r in by_device.items()},
    'gpu_kernel_busy_ms': {str(d): sum(end-start for start, end in ranges)/1e6 for d, ranges in unions.items()},
    'simultaneous_kernel_busy_ms': overlap/1e6,
    'sample_pairs': min(20, len(samples)),
}
(output / 'nsys_overlap.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False))
