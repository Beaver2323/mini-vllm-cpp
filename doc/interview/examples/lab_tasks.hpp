#ifndef ZYF_INTERVIEW_LAB_TASKS_HPP
#define ZYF_INTERVIEW_LAB_TASKS_HPP
// 只在此学习副本补全，不修改 mini_vllm/ 下的引擎。
#include "mini_vllm/scheduler.hpp"
#include <stdexcept>
#include <vector>
namespace interview_lab {
inline bool sample_ready(const mini_vllm::Sequence& sequence, std::size_t scheduled) {
    // TODO 1：检查 scheduled 合法，并判断本轮是否完成所有已知输入。
    throw std::logic_error("TODO 1: sample_ready");
}
inline std::size_t physical_slot(const std::vector<int>& table,
                                 std::size_t position, std::size_t page_size) {
    // TODO 2：逻辑页 -> 物理页 -> 页内槽；拒绝非法参数。
    throw std::logic_error("TODO 2: physical_slot");
}
inline std::vector<std::size_t> sample_rows(const mini_vllm::SchedulerOutput& output,
                                           const std::vector<std::size_t>& starts) {
    // TODO 3：检查请求边界，用 TODO 1 选择真正可采样的 packed 行。
    throw std::logic_error("TODO 3: sample_rows");
}
inline void restore_after_handoff(mini_vllm::Sequence& target,
                                   const mini_vllm::Sequence& source) {
    // TODO 4：KV 已复制且参数一致的前提下，恢复 Prompt computed 和首输出。
    // 此函数只练习 CPU 记账，不执行真正跨 GPU 迁移。
    throw std::logic_error("TODO 4: restore_after_handoff");
}
} // namespace interview_lab
#endif
