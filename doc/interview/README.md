# 面向面试的理解与复述路线

这组材料按你的顺序补充：先读固定版本 vLLM 对照和项目面试手册，再用代码补全、故障定位和性能分析检验理解。原来的任务01—11仍负责详细源码精读；这里负责组织回答和追问。

## 1. 基础五篇的阅读顺序

| 顺序 | 文档 | 先记住什么 | 再用什么检查 |
| --- | --- | --- | --- |
| 1 | [固定版本 vLLM 源码对照](01_vllm_source_map_zh.md) | 调度、KV、Runner、Attention职责与实现差异 | 18 Token请求沿固定提交逐函数跟踪 |
| 2 | [项目口述、追问与证据](02_project_defense_zh.md) | 30秒/3分钟介绍、12组高频卡、最小数字卡 | 每个结论指出源码与实验边界 |
| 3 | [四个代码补全实验](03_coding_labs_zh.md) | 采样资格、slot、行映射、PD恢复 | 补全独立学习副本，通过CPU检查 |
| 4 | [四个故障定位案例](04_debug_cases_zh.md) | 现象→首个分歧→根因→修复→回归 | 运行明确标注的教学错误与GDB观察 |
| 5 | [性能分析实操](05_profiling_lab_zh.md) | 实验分组、计时边界、trace与因果判断 | 从原始CSV重算数字、手算真实重叠区间 |

每篇采用短答、展开、源码、追问的层次。短答适合复述；展开部分帮助处理面试官改变输入或条件后的问题。

### 新增：三篇详细代码课程，按 06 → 07 → 08 学习

这三篇对应你要求的“完整执行过程、PyTorch/CUDA 逐段对照、从测试反推正确性”。每篇都有实际代码摘录、调用点、手算、参考答案和已运行的学习材料。可以分多次阅读，不必一次背完。

| 顺序 | 详细课程 | 完成后应该能做什么 |
| --- | --- | --- |
| 06 | [两个请求从入队到回收](06_request_walkthrough_zh.md) | 手推四轮状态、五行 Packed 元数据，解释 CPU/GPU 边界 |
| 07 | [PyTorch 与 C++/CUDA 逐段对照](07_pytorch_cuda_bridge_zh.md) | 从 Tensor 公式算到页内元素地址，读懂 Attention 与 LM Head |
| 08 | [从测试反推正确性](08_tests_as_spec_zh.md) | 解释真实断言的依据、能检测的错误和覆盖边界 |

06 附同一示例的 CPU 控制面与真实 CUDA 微型模型记录；07 附四组 CPU PyTorch 算子实验；08 附在临时副本中注入三个错误、触发现有断言的实测记录。本轮验证与命令见 [新增课程验证记录](examples/deepening_validation.md)。

建议先完成入门路线前三节，再读 06；遇到 Tensor 或 kernel 不熟悉时读 07；最后读 08，用测试检验前面的理解。面试复述继续配合手册 02。

## 2. 一次学习只完成一个小闭环

1. 看一道面试问题，先用自己的话答30秒。
2. 阅读对应原理和代码，纠正一个最不确定的点。
3. 遮住答案，写一条公式或一组状态变化。
4. 用实验、原始记录或源码核对。
5. 再答一次，补上实现证据与边界。

掌握标准不是背完页数，而是改变页号、Prompt长度或R之后仍能推导。长路径和精确行号可以查，因果关系和状态含义需要自己讲清楚。

## 3. 第一轮需要掌握的最小集合

| 必答问题 | 对应手册 |
| --- | --- |
| 项目是什么，基于什么，主要能力是什么 | 手册02的30秒介绍 |
| 为什么缓存KV，生成ID为何不立刻有KV | 手册02卡01/02，实验L1 |
| 分页地址与packed位置如何区分 | 手册01第8/9节，实验L2/L3 |
| 采样行和Prefix分别省了什么 | 手册02卡05/06，手册05第5/6节 |
| Graph与融合为什么不保证加速 | 手册02卡07/08，手册05 |
| PD恢复什么，当前为什么更慢 | 手册02卡10/11，实验L4 |
| 本项目与固定vLLM版本哪里不同 | 手册01第4/6/11/13节 |

第一轮不用把全部vLLM模型、通信后端和配置选项都背下来。把这里的普通生成路径讲清楚后，再扩展自己岗位相关的方向。

## 4. 可运行材料

全部CPU教学实验与统计分析均已在本机验证，记录见 [verified_output.txt](examples/verified_output.txt)；GDB观察见 [gdb_verified_output.txt](examples/gdb_verified_output.txt)。

```bash
cd /home/users/zyf/zyf_llm.c/llm.c
c++ -std=c++17 -O0 -g -gdwarf-4 -I. -DLAB_USE_SOLUTION \
  doc/interview/examples/lab_checks.cpp -o /tmp/zyf_interview_lab_answers
/tmp/zyf_interview_lab_answers all

c++ -std=c++17 -O0 -g -gdwarf-4 -I. \
  doc/interview/examples/fault_cases.cpp -o /tmp/zyf_interview_faults
/tmp/zyf_interview_faults

conda run -p /home/miniconda3/envs/zyf1 python \
  doc/interview/examples/analyze_evidence.py
```

题目文件保留TODO，未补完时失败是预期行为；上面的 `LAB_USE_SOLUTION` 明确运行参考答案。教学错误程序成功退出表示错误已被检测。

当前验证范围：CPU记账与映射、教学反例、原始数据分析、已有SQLite重叠复核；没有因为编写这组文档重新运行全部GPU基准，也没有安装运行vLLM。

## 5. 与原有资料配合

- 不理解模型生成原理：回到 [PyTorch入门六节](../from_pytorch/README.md)。
- 想逐段读完整实现：使用 [任务01—11源码索引](../paged_inference_learning_zh.md)。
- 需要核对面试数字：打开 [任务09—11实测记录](../../benchmark/results/task09_11/README.md)。
- 需要确认外部代码版本：查看 [固定来源清单](references/sources.json)，vLLM固定为v0.10.2提交`01efc7ef781391e744ed08c3292817a773d654e6`。

个人经历按真实参与范围表达，教学案例与已验证的项目能力分开。先把能查到证据的回答讲稳，再谈尚未实现的改进设计。
