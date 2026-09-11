# 详细代码课程 06—08：本轮验证记录

日期：2026-09-11。项目实现基线：`9903541`；本轮新增学习文档与示例，不修改引擎实现。

## 1. 已运行的项目

| 检查 | 结果 | 完整记录 |
| --- | --- | --- |
| 请求执行示例：CPU 控制面 | 四轮调度、计数、页回收与复用通过 | [CPU 输出](request_walkthrough_cpu.txt) |
| 同一请求示例：真实 CUDA Runner | 两层微型模型 FP32/Eager 运行通过 | [CUDA 输出](request_walkthrough_cuda.txt) |
| CPU / CUDA 元数据对照 | N/R、positions、context、slot、页表、队列末态一致 | 上述两份记录 |
| PyTorch 四组算子语义实验 | QKV、KV、Attention、LM Head 检查全部通过 | [PyTorch 输出](pytorch_operator_bridge_output.txt) |
| 原控制面测试 | 未修改源码的基线通过 | [错误注入记录首行](mutation_demo_output.txt) |
| 三个受控错误 | 每个变体编译成功，分别触发原有断言 | [错误注入完整输出](mutation_demo_output.txt) |
| GDB 第 3 轮断点 | computed=17，offset=0，position=17，pages=[0,1] | 下方命令与摘录 |

Python 使用 `zyf1`：`/home/miniconda3/envs/zyf1/bin/python`，Python 3.8 / PyTorch 2.4.1+cu121。PyTorch 实验实际在 CPU 上执行，精度为 float64。

CUDA 示例使用本机 RTX 3090，编译目标 `sm_86`。模型配置为 max_context=32、vocab=64、layers=2、heads=4、channels=32，固定公式生成未训练权重。不需要 `gpt2_124M.bin`。

## 2. 复现命令

在仓库根目录执行，不添加 `-DNDEBUG`，Python 不添加 `-O`：

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
conda activate zyf1

c++ -std=c++17 -O0 -g -gdwarf-4 -I. \
  doc/interview/examples/request_walkthrough.cpp -o /tmp/zyf_request_walkthrough
/tmp/zyf_request_walkthrough

/usr/local/cuda/bin/nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 \
  -DWALK_CUDA -I. doc/interview/examples/request_walkthrough.cpp \
  mini_vllm/cuda/gpt2_cuda_model_runner.cu mini_vllm/cuda/paged_attention.cu \
  -lcublas -o /tmp/zyf_request_walkthrough_cuda
/tmp/zyf_request_walkthrough_cuda

python doc/interview/examples/pytorch_operator_bridge.py --case all
python doc/interview/examples/mutation_demo.py

c++ -std=c++17 -O0 -g -gdwarf-4 \
  dev/test_mini_vllm_control_plane.cpp -o /tmp/zyf_deepening_control_tests
/tmp/zyf_deepening_control_tests
```

错误注入脚本成功退出意味着三个错误均被检测；内部变体的断言失败是预期结果。临时路径在保存的错误信息中替换为 `<临时目录>`，方便阅读。

## 3. CUDA 元数据与输出核对

| 轮次 | N | R | 元数据 H2D bytes | GPU 返回的 item 标记 |
| --- | --- | --- | --- | --- |
| 1 | 16 | 0 | 384 | `[-1]` |
| 2 | 16 | 1 | 388 | `[4,-1]` |
| 3 | 5 | 2 | 128 | `[25,21]` |
| 4 | 2 | 2 | 56 | `[0,21]` |

公式为 `sizeof(int) × (4N + 2N + R)`：四个逐行基础数组、每行宽度为2的页表、R个采样索引。不包含权重或输出 ID 复制。

两个版本使用同一请求长度与调度条件，但 CPU 版本用教学采样规则，输出 ID 与 GPU 不同是预期行为。元数据中的 Token ID 会因此不同；position、slot、page、count 等不依赖具体输出值的部分已逐轮比较一致。

## 4. GDB 观察实测

前两轮分别构造16个 Token 行，因此忽略前32次 position 构造断点，可停在第3轮第0行。以下为本轮运行命令：

```bash
gdb -q -batch /tmp/zyf_request_walkthrough \
  -ex 'set pagination off' \
  -ex 'break mini_vllm/model_input.hpp:64' \
  -ex 'ignore 1 32' -ex run \
  -ex 'print item_index' -ex 'print offset' \
  -ex 'print sequence.num_computed_tokens_' \
  -ex next -ex 'print position' -ex 'print sequence.block_table_' \
  -ex 'disable 1' -ex continue
```

关键输出摘录：

```text
Breakpoint 1, mini_vllm::prepare_packed_model_input (...) at ./mini_vllm/model_input.hpp:65
65                  sequence.num_computed_tokens() + offset;
$1 = 0
$2 = 0
$3 = 17
66              if (position >= sequence.num_tokens() ||
$4 = 17
$5 = std::vector of length 2, capacity 2 = {0, 1}
```

对应顺序：item_index=0、offset=0、computed=17；单步完成赋值后 position=17；A 页表为 `[0,1]`。程序随后正常结束。GDB 将第64行断点定位到有机器指令的第65行，是本次编译下的正常行号映射。

## 5. 文档与源码一致性

三篇引用的56处源码/记录摘录按当前文件的实际行范围生成，并逐段比对。来源文件 SHA-256、行数和摘录范围保存在 [来源清单](../references/deepening_sources.json)。未来代码变更后，可据此发现文档与实现的版本差异。

本轮没有重新运行全部 GPT-2 124M GPU 回归、全部精度组合或性能 benchmark。课程08引用的这部分证据来自已有 [任务09—11记录](../../../benchmark/results/task09_11/validation.txt)。没有安装或执行外部 vLLM。
