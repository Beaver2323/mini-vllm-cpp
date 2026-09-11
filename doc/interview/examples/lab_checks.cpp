// 使用真实 Sequence/BlockManager/ModelInput，全部实验仅运行 CPU。
#include "mini_vllm/model_input.hpp"
#ifdef LAB_USE_SOLUTION
#include "lab_solutions.hpp"
#else
#include "lab_tasks.hpp"
#endif
#include <functional>
#include <iostream>
#include <string>
using namespace mini_vllm;
using namespace interview_lab;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void rejects(const std::function<void()>& action) {
    bool thrown = false;
    try { action(); } catch (const std::exception&) { thrown = true; }
    require(thrown, "invalid input must be rejected");
}
static void lab1() {
    Sequence s(1, {10,11,12,13,14}, {3,-1,true});
    require(!sample_ready(s,3), "partial Prompt must not sample");
    s.mark_computed(3);
    require(sample_ready(s,2), "final Prompt chunk must sample");
    s.mark_computed(2);s.append_token(20);
    require(s.num_computed_tokens()==5 && s.num_tokens()==6, "sample ID has no KV yet");
    require(sample_ready(s,1), "Decode input must sample");
    rejects([&]{sample_ready(s,0);});rejects([&]{sample_ready(s,2);});
    std::cout << "L1 PASS: partial=false final=true decode=true computed=5 total=6\n";
}
static void lab2() {
    const std::vector<int> table{5,2,9};
    require(physical_slot(table,15,16)==95, "last position in first page");
    require(physical_slot(table,16,16)==32, "first position in second page");
    require(physical_slot(table,17,16)==33, "second-page offset");
    require(physical_slot(table,32,16)==144, "third logical page");
    rejects([&]{physical_slot(table,48,16);});
    rejects([&]{physical_slot(table,1,0);});
    rejects([&]{physical_slot({-1},0,16);});
    std::cout << "L2 PASS: positions=[15,16,17,32] slots=[95,32,33,144]\n";
}
static void lab3() {
    BlockManager blocks(8,16);
    auto a=std::make_shared<Sequence>(1,std::vector<int>(17,10),SamplingParams{4,-1,true});
    auto b=std::make_shared<Sequence>(2,std::vector<int>{20,21,22},SamplingParams{4,-1,true});
    a->mark_computed(17);a->append_token(30);
    require(blocks.ensure_capacity(*a,18) && blocks.ensure_capacity(*b,3), "allocate pages");
    SchedulerOutput output{{{a,ExecutionPhase::Decode,1},{b,ExecutionPhase::Prefill,3}},4};
    auto packed=prepare_packed_model_input(output,blocks,64,4,8);
    require(packed.positions==std::vector<int>({17,0,1,2}), "absolute positions");
    require(sample_rows(output,packed.query_start_locations)==std::vector<std::size_t>({0,3}), "mixed sample rows");
    output.items[1].num_scheduled_tokens=2;output.num_batched_tokens=3;
    packed=prepare_packed_model_input(output,blocks,64,4,8);
    require(sample_rows(output,packed.query_start_locations)==std::vector<std::size_t>({0}), "partial B must be omitted");
    SchedulerOutput partial{{{b,ExecutionPhase::Prefill,2}},2};
    require(sample_rows(partial,{0,2}).empty(), "R=0 is valid");
    rejects([&]{sample_rows(output,{0,2,3});});
    blocks.release(*a);blocks.release(*b);blocks.validate();
    require(blocks.num_free_blocks()==8,"all pages returned");
    std::cout << "L3 PASS: positions=[17,0,1,2] rows=[0,3] partial_rows=[0] R0=[]\n";
}
static void lab4() {
    Sequence source(7,std::vector<int>(17,10),{4,-1,true});
    Sequence target(7,std::vector<int>(17,10),{4,-1,true});
    source.mark_computed(17);source.append_token(42);
    restore_after_handoff(target,source);
    require(target.num_computed_tokens()==17 && target.num_tokens()==18 &&
            target.pending_tokens()==1 && target.token_ids().back()==42,"D must execute first generated ID");
    target.mark_computed(1);target.append_token(43);
    require(target.num_computed_tokens()==18 && target.num_tokens()==19,"first D commit");
    Sequence wrong(7,std::vector<int>(17,11),{4,-1,true});
    rejects([&]{restore_after_handoff(wrong,source);});
    require(wrong.num_computed_tokens()==0,"reject before changing target");
    rejects([&]{restore_after_handoff(target,source);});
    std::cout << "L4 PASS: handoff=17/18 first_decode=18/19 invalid_prefix=rejected\n";
}
int main(int argc,char** argv) {
    try {
        const std::string which=argc>1?argv[1]:"all";
        if (argc>2 || (which!="all" && which!="1" && which!="2" && which!="3" && which!="4"))
            throw std::invalid_argument("usage: lab_checks [all|1|2|3|4]");
        if(which=="all"||which=="1")lab1();
        if(which=="all"||which=="2")lab2();
        if(which=="all"||which=="3")lab3();
        if(which=="all"||which=="4")lab4();
    } catch(const std::exception& e) {std::cerr << "实验未通过: " << e.what() << '\n';return 1;}
}
