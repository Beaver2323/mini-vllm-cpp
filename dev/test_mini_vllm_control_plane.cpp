#include "../mini_vllm/scheduler.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace mini_vllm;

static std::shared_ptr<Sequence> request(std::uint64_t id, std::size_t prompt_len,
                                         std::size_t max_new_tokens) {
    std::vector<int> prompt(prompt_len);
    for (std::size_t i = 0; i < prompt_len; ++i) prompt[i] = int(id * 100 + i);
    return std::make_shared<Sequence>(
        id, std::move(prompt), SamplingParams{max_new_tokens, -1, false});
}

static void test_block_allocation_release_and_reuse() {
    BlockManager manager(6, 4);
    auto a = request(1, 5, 1);
    auto b = request(2, 4, 1);
    assert(manager.ensure_capacity(*a, 5));
    assert(manager.ensure_capacity(*b, 4));
    assert(a->block_table().size() == 2);
    assert(manager.block_id_for_token(*a, 4) == a->block_table()[1]);
    assert(manager.slot_for_token(4) == 0);
    const std::vector<int> released = a->block_table();
    manager.release(*a);
    assert(manager.num_free_blocks() == 5);

    // Consume every free block. FIFO allocation need not reuse the most recently
    // released block first, but all released capacity must eventually be reusable.
    auto c = request(3, 20, 1);
    assert(manager.ensure_capacity(*c, 20));
    assert(c->block_table().size() == 5);
    assert(c->block_table()[3] == released[1]);
    assert(c->block_table()[4] == released[0]);
    manager.validate();

    bool rejected_double_free = false;
    try { manager.release(*a); }
    catch (const std::logic_error&) { rejected_double_free = true; }
    assert(rejected_double_free);
}

static void test_chunked_prefill() {
    BlockManager manager(8, 4);
    Scheduler scheduler({1, 4}, manager);
    auto a = request(10, 9, 1);
    scheduler.add(a);

    auto first = scheduler.schedule();
    assert(first.items.size() == 1 && first.items[0].num_scheduled_tokens == 4);
    assert(first.items[0].phase == ExecutionPhase::Prefill);
    scheduler.commit(first, {-1});

    auto second = scheduler.schedule();
    assert(second.items[0].num_scheduled_tokens == 4);
    scheduler.commit(second, {-1});

    auto third = scheduler.schedule();
    assert(third.items[0].num_scheduled_tokens == 1);
    scheduler.commit(third, {777});
    assert(a->is_finished());
    assert(scheduler.is_finished());
    assert(manager.num_free_blocks() == manager.num_blocks());
}

static void test_oom_is_atomic_and_eos_releases_blocks() {
    BlockManager manager(2, 4);
    auto owner = request(30, 8, 1);
    auto blocked = request(31, 1, 1);
    assert(manager.ensure_capacity(*owner, 8));
    assert(!manager.ensure_capacity(*blocked, 1));
    assert(blocked->block_table().empty());
    assert(manager.num_free_blocks() == 0);
    manager.release(*owner);

    Scheduler scheduler({1, 4}, manager);
    auto eos_request = std::make_shared<Sequence>(
        32, std::vector<int>{1, 2}, SamplingParams{10, 42, false});
    scheduler.add(eos_request);
    auto output = scheduler.schedule();
    scheduler.commit(output, {42});
    assert(eos_request->is_finished());
    assert(manager.num_free_blocks() == manager.num_blocks());
}

static void test_continuous_admission_and_retirement() {
    BlockManager manager(12, 4);
    Scheduler scheduler({2, 5}, manager);
    auto a = request(20, 5, 2);
    auto b = request(21, 2, 1);
    auto c = request(22, 3, 1);
    scheduler.add(a);
    scheduler.add(b);
    scheduler.add(c);

    auto step1 = scheduler.schedule();
    assert(step1.items.size() == 1 && step1.num_batched_tokens == 5);
    scheduler.commit(step1, {900});
    assert(scheduler.num_running() == 1 && scheduler.num_waiting() == 2);

    auto step2 = scheduler.schedule();
    assert(step2.items.size() == 2);
    assert(step2.items[0].sequence == a && step2.items[0].phase == ExecutionPhase::Decode);
    assert(step2.items[1].sequence == b && step2.items[1].phase == ExecutionPhase::Prefill);
    scheduler.commit(step2, {901, 902});
    assert(a->is_finished() && b->is_finished());
    assert(scheduler.num_running() == 0 && scheduler.num_waiting() == 1);

    auto step3 = scheduler.schedule();
    assert(step3.items.size() == 1 && step3.items[0].sequence == c);
    scheduler.commit(step3, {903});
    assert(c->is_finished() && scheduler.is_finished());
    assert(manager.num_free_blocks() == manager.num_blocks());
}

int main() {
    test_block_allocation_release_and_reuse();
    test_chunked_prefill();
    test_oom_is_atomic_and_eos_releases_blocks();
    test_continuous_admission_and_retirement();
    std::cout << "mini-vLLM control-plane tests passed: block reuse, atomic OOM, EOS, "
                 "chunked prefill, continuous admission/retirement\n";
}
