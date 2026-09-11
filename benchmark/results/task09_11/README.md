# 任务 09—11：本机实测结果

2026-09-11 在本机两张 NVIDIA RTX 3090（各 24 GiB）执行。CUDA 12.5、sm_86、
OMP_NUM_THREADS=8、GPT-2 124M、FP16。PD 使用 GPU 0 Prefill / GPU 1 Decode；其余测试用 GPU 0。

## 版本与测量口径

- [environment.json](environment.json) 记录 GPU、驱动、CUDA、拓扑、checkpoint 与实现源码 SHA256。
- 本次基于 `94996b4` 后的未提交实现测量，因此旧 Benchmark 内嵌的 git_commit 是父提交，
  **实际代码以 environment.json 中的源码哈希为准**；不可把新结果归到父提交的代码上。
- 每种模式一次预热，正式重复 7 次；下面使用中位数。模型加载和 Buffer 构造不计时。
- 所有正式生成结果均通过一致性检查；采样行/Prefix 的独立参考是 CPU 完整前缀，PD 同时
  对齐单卡和 CPU。FP16 数值误差按测试阈值检查，没有声称与 FP32 逐 bit 相同。
- 这是固定、少量 Token ID 输入的实验，无网络/分词器/真实服务请求到达过程。

## 1. 采样行裁剪

4 请求的 Prompt 为 8/16/24/32，各生成 4 Token；最大请求数 4，Token Budget 64，Context 64，
KV Pool 16 页。FP16，Fusion 关闭。Graph 分组独立对比，默认裁剪开启，`--full-logits` 恢复全行。

| 执行方式 | LM Head | 整批耗时 ms | 输出 tok/s | TTFT P50 ms | 投影总行数 | 激活缓冲区 MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 全行 | 8.254 | 1938.5 | 1.775 | 92 | 13.781 |
| Eager | 裁剪 | 8.038 | 1990.7 | 1.718 | 16 | 2.180 |
| Graph | 全行 | 5.314 | 3010.6 | 1.154 | 92 | 13.781 |
| Graph | 裁剪 | 5.227 | 3061.2 | 1.110 | 16 | 2.180 |

裁剪使投影行数 92 → 16。这里 92 = 80 个 Prompt 输入 + 12 个 Decode 输入；16 是四个请求的
全部输出 Token 数。Graph 内延迟中位数减少约 1.7%，Eager 约 2.6%。
小幅时间差可能受运行噪声影响，不能宣称所有场景稳定获得同样百分比。持久 logits 容量减少
更明确；表中的 activation_bytes 是 Runner 统计的激活 Buffer 总和，**不是整张 GPU 的显存占用**。

原始 JSON/CSV：`rows_full.*`、`rows_pruned.*`、`rows_full_graph.*`、`rows_pruned_graph.*`。
旧任务 07 数据是另一轮运行，不用于这里的跨版本因果比较。

## 2. Prefix Cache

Batch=1、Budget=272、Context=272、KV Pool=40 页，FP16、Fusion/Graph 关闭、采样行裁剪开启。
seed 长度为 prefix+1，测量 Prompt 为 prefix+2，共生成 4 Token。每个模式都清空缓存；hit 模式
另外运行 seed。清理、seed、CPU 参考在测量窗口外。

| 共享前缀 Token | off TTFT ms | miss TTFT ms | hit TTFT ms | hit 相对 miss 的 TTFT 降幅 | 完整窗口输入量 off→hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.251 | 1.244 | 1.119 | 10.1% | 21 → 5 |
| 64 | 1.649 | 1.649 | 1.306 | 20.8% | 69 → 5 |
| 128 | 2.169 | 2.178 | 1.317 | 39.5% | 133 → 5 |
| 256 | 4.058 | 4.057 | 1.541 | 62.0% | 261 → 5 |

每组检查 hit_blocks 与 scheduled_tokens，且四个输出 Token 都与 CPU 完整前缀参考一致。
调度量下降发生在 Prefill，Decode 仍访问同样长度的历史 KV；完整生成吞吐不是按前缀命中比例
等比例提高。[prefix.csv](prefix.csv) 保存 TTFT、TPOT、总时间、吞吐、命中块数和实际输入量。

## 3. 双 GPU PD 的功能与成本

3 个 Prompt 长度 17/33/49，各生成 8 Token。单卡最大 3 请求、Budget=32、KV Pool=24 页；
PD 的 P Budget=32、P Pool=8 页、D 最大 3 请求、D Pool=16 页；Context 都是 64。
FP16、Fusion 关闭、采样行裁剪开启。PD 的迁移、同步和主机线程开销全部计入总时间。

| 模式 | 单卡整批 ms | PD 整批 ms | PD 交接合计 ms | PD/单卡耗时比 |
| --- | ---: | ---: | ---: | ---: |
| Eager | 11.640 | 25.906 | 7.205 | 2.23× |
| Graph | 9.876 | 18.493 | 7.018 | 1.87× |

本次 PD 更慢。它使用两份权重、同步 host staging，且每次交接重新分配 pinned buffer，
短请求中这些开销明显。D Pool 预留完整生成上限以避免扩页死锁，P/D 计算按轮等待；当前没有
传输与计算重叠、长期 Worker 线程或动态负载均衡。

每轮三个请求共搬 9 页，K+V 载荷 5,308,416 字节（5.0625 MiB），D2H+H2D 的主机传输量
10,616,832 字节。`transfer_ms` 包括 staging 分配/释放、设备切换、按页拷贝和同步，不能直接
拿它与单个连续 memcpy 的理想带宽对比。Payload 含最后一页未用槽位。

每次固定工作负载产生 4 轮 P/D 并发提交。其计数是提交行为，不等于速度提升。
[pd.csv](pd.csv)、[pd_graph.csv](pd_graph.csv) 保存每次测量。单卡对照用于观察功能拆分成本；
判断双卡资源效率还需要“两卡各跑完整请求”的数据并行对照，本轮没有据此宣称多卡加速。

## 4. 正确性和内存检查

- 采样行测试固定 N=4、R=0/1/2/1/0；Graph 三种形状和动态索引复用均通过。
  与完整行 Eager 比较的最大误差：FP16 0.000267029，FP32 0.0000305176。
- FP32、FP16、FP16 + Fusion + Graph、完整 logits 回归通过；Argmax mismatch=0。
- Prefix 模型级用例命中 16 Token 后仅调度 2 Token，最大词表误差 0.104347 < 0.2。
- PD Eager/Graph：不同物理页号迁移后连续三步整行 logits 对齐；Prompt 1/16/17/31/32/33、
  源/目标页回收、首 Token 结束、受限 D Pool 排队全部通过。Greedy 与 CPU/单卡一致。
- CUDA Compute Sanitizer memcheck：PD + Graph、采样行 + Graph 均为 0 errors。
- CPU 控制面测试通过，覆盖分页、atomic OOM、EOS、Chunked Prefill、动态准入/退出、Prefix/LRU。

见 [完整验证输出](validation.txt)、[PD Eager 验证](pd_eager_validation.txt)、
[PD 显存检查](sanitizer_pd.txt)、[采样行显存检查](sanitizer_rows.txt)。

## 5. 复现

从仓库根目录执行，checkpoint 使用 `gpt2_124M.bin`：

```bash
conda activate zyf1
bash scripts/validate_pd_cuda.sh benchmark/results/pd_reproduction
```

脚本构建指定架构，记录环境/源码哈希，执行正确性、四组采样行对照、Prefix 三组对照、PD
单卡对照和 memcheck。学习时可以只运行相应任务文档中的单个命令，无需每次全量运行。

下面是 profiler 复现入口，采集运行与正式性能测量分开：

```bash
OMP_NUM_THREADS=8 nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/pd_overlap \
  ./benchmark_gpt2_pd_serving /tmp/pd_profile.csv --cuda-graph
```


## 6. 两卡是否真的同时计算

独立 Nsight 运行用 `--cuda-graph-trace=node` 采集 Graph 内部 Kernel。默认 graph 粒度只记录整张
图，不能直接拿 Kernel 表判断图内执行。GPU 0 记录 26,034 个 Kernel，GPU 1 记录 18,009 个；
两端 Kernel 区间的交集并集时间为 **0.294998 ms**。这是整次带 profiler 运行的累计值，含预热
和七轮实验，显示存在真实重叠，但重叠量很小，不证明获得了吞吐收益。node 级采集自身有开销，
不能用该比例推断未采集时的重叠比例。

[nsys_overlap.json](nsys_overlap.json) 保存汇总和 SQLite 哈希，
[nsys_overlap_samples.csv](nsys_overlap_samples.csv) 保存前 20 对跨卡重叠 Kernel 的原始时间、
Stream 和名称。完整本机轨迹位于 `/tmp/zyf_pd_overlap_nodes.nsys-rep`，SQLite 导出位于
`/tmp/zyf_pd_overlap_nodes.sqlite`，未将大型二进制轨迹提交到 Git。

```bash
nsys export --type=sqlite --force-overwrite=true \
  --output=/tmp/pd_overlap.sqlite /tmp/pd_overlap.nsys-rep
python scripts/summarize_pd_overlap.py /tmp/pd_overlap.sqlite benchmark/results/pd_reproduction
```

脚本先合并各设备的 Kernel 时间区间，再求两集合交集，避免把同一时间的多个重叠事件重复累加。
