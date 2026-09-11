// 四个人为构造的教学错误：只验证 CPU 逻辑，不执行错误 CUDA kernel。
#include "mini_vllm/model_input.hpp"
#include "lab_solutions.hpp"
#include <iostream>
#include <map>
#include <stdexcept>
using namespace mini_vllm;
static void detected(bool mismatch,const char* message) {
    if(!mismatch)throw std::runtime_error("教学错误未被检测");
    std::cout << message << '\n';
}
int main() {
    try {
        // F1：绕过页表，把逻辑位置当作物理 slot。
        const std::vector<int> table{5,2};
        const auto expected_slot=interview_lab::physical_slot(table,17,16);
        const auto bad_slot=(17/16)*16+17%16;
        detected(expected_slot!=static_cast<std::size_t>(bad_slot),
                 "F1 已检测: position=17 错误slot=17 正确slot=33");
        // F2：新请求加入后，错误地按本轮 offset 重置已有请求的 position。
        BlockManager blocks(8,16);
        auto a=std::make_shared<Sequence>(1,std::vector<int>(17,10),SamplingParams{4,-1,true});
        auto b=std::make_shared<Sequence>(2,std::vector<int>{20,21,22},SamplingParams{4,-1,true});
        a->mark_computed(17);a->append_token(30);
        if(!blocks.ensure_capacity(*a,18)||!blocks.ensure_capacity(*b,3))throw std::runtime_error("allocation");
        SchedulerOutput output{{{a,ExecutionPhase::Decode,1},{b,ExecutionPhase::Prefill,3}},4};
        const auto input=prepare_packed_model_input(output,blocks,64,4,8);
        const std::vector<int> wrong_positions{0,0,1,2};
        detected(input.positions!=wrong_positions,
                 "F2 已检测: 错误positions=[0,0,1,2] 正确=[17,0,1,2]");
        blocks.release(*a);blocks.release(*b);blocks.validate();
        // F3：用 map 模拟把动态采样行误当成静态图缓存内容。
        // 此处没有运行真正 CUDA Graph，检验的是元数据必须更新的条件。
        std::map<std::pair<int,int>,std::vector<int>> bad_cached_rows;
        const auto key=std::make_pair(4,1);
        bad_cached_rows.emplace(key,std::vector<int>{3});
        const std::vector<int> current_rows{1};
        detected(bad_cached_rows.at(key)!=current_rows,
                 "F3 已检测: 同(N,R)=(4,1) 旧rows=[3] 当前必须=[1]");
        // F4：交接后误认为首生成 ID 也已有 KV，多推进一次 computed。
        Sequence source(7,std::vector<int>(17,10),{4,-1,true});
        Sequence target(7,std::vector<int>(17,10),{4,-1,true});
        source.mark_computed(17);source.append_token(42);
        interview_lab::restore_after_handoff(target,source);
        const auto expected_pending=target.pending_tokens();
        target.mark_computed(1); // 故意错误：并没有执行首输出的模型前向。
        detected(target.pending_tokens()!=expected_pending,
                 "F4 已检测: 错误computed/total=18/18 正确=17/18 pending必须为1");
        std::cout << "四个教学错误均被定位条件检测；生产引擎未修改。\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
