// Independent dense reference for the CPU paged attention implementation.
#include "../paged_kv_cache.hpp"
#include <algorithm>
#include <limits>

int main() {
    constexpr int B = 2, L = 2, NH = 2, HS = 4, C = NH * HS, T = 33;
    constexpr int blocks = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    KVCachePool pool(B * blocks, L, NH, HS);
    PageTable table(B, blocks);
    // Reverse the allocation order to ensure physical pages are not logical pages.
    std::reverse(pool.free_pages.begin(), pool.free_pages.end());
    std::vector<float> history(L * B * T * 3 * C);
    for (size_t i = 0; i < history.size(); ++i)
        history[i] = std::sin(float(i) * 0.137f);
    std::vector<float> qkv(B * 3 * C), out(B * C), scratch(B * NH * T);
    double max_error = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            if (t % PAGE_SIZE == 0)
                table.block_tables[b * blocks + t / PAGE_SIZE] = pool.allocate_page();
            table.context_lengths[b] = t + 1;
        }
        for (int l = 0; l < L; ++l) {
            for (int b = 0; b < B; ++b)
                std::copy_n(history.data() + ((l * B + b) * T + t) * 3 * C,
                            3 * C, qkv.data() + b * 3 * C);
            paged_attention_forward(out.data(), qkv.data(), &pool, &table,
                                    scratch.data(), l, B, t + 1, C, NH, T);
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < NH; ++h) {
                    std::vector<double> scores(t + 1);
                    double max_score = -std::numeric_limits<double>::infinity();
                    for (int s = 0; s <= t; ++s) {
                        double score = 0.0;
                        for (int d = 0; d < HS; ++d)
                            score += double(qkv[b * 3 * C + h * HS + d]) *
                                history[((l * B + b) * T + s) * 3 * C + C + h * HS + d];
                        scores[s] = score / std::sqrt(double(HS));
                        max_score = std::max(max_score, scores[s]);
                    }
                    double sum = 0.0;
                    for (double& score : scores) { score = std::exp(score - max_score); sum += score; }
                    for (int d = 0; d < HS; ++d) {
                        double expected = 0.0;
                        for (int s = 0; s <= t; ++s)
                            expected += scores[s] / sum *
                                history[((l * B + b) * T + s) * 3 * C + 2 * C + h * HS + d];
                        double actual = out[b * C + h * HS + d];
                        if (!std::isfinite(actual)) return 1;
                        max_error = std::max(max_error, std::abs(actual - expected));
                    }
                }
            }
        }
    }
    std::cout << "B=2 L=2 NH=2 HS=4 lengths=1..33, reversed/interleaved pages, max_abs_error="
              << max_error << "\n";
    return max_error < 1e-5 ? 0 : 1;
}
