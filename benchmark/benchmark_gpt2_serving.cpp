#include "../mini_vllm/gpt2_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MINI_VLLM_GIT_COMMIT
#define MINI_VLLM_GIT_COMMIT "unknown"
#endif
#ifndef MINI_VLLM_BUILD_FLAGS
#define MINI_VLLM_BUILD_FLAGS "unknown"
#endif

using namespace mini_vllm;
using Clock = std::chrono::steady_clock;

struct WorkloadRequest {
    std::uint64_t request_id;
    std::vector<int> prompt_tokens;
    std::size_t max_new_tokens;
};

struct RequestMetrics {
    std::uint64_t request_id;
    double ttft_ms;
    double tpot_ms;
    double latency_ms;
};

struct RunMetrics {
    std::string mode;
    int repeat;
    double total_seconds;
    std::size_t output_tokens;
    double output_tokens_per_second;
    double ttft_p50_ms;
    double ttft_p95_ms;
    double tpot_p50_ms;
    double tpot_p95_ms;
    double latency_p50_ms;
    double latency_p95_ms;
    std::size_t peak_kv_blocks;
    std::vector<RequestMetrics> requests;
    std::vector<std::vector<int>> outputs;
};

struct Options {
    std::string checkpoint = "gpt2_124M.bin";
    std::string json_path = "benchmark/results/gpt2_cpu.json";
    std::string csv_path = "benchmark/results/gpt2_cpu.csv";
    int repeats = 3;
    std::size_t max_new_tokens = 4;
};

struct ModeSummary {
    std::string mode;
    double total_seconds;
    double output_tokens_per_second;
    double ttft_p50_ms;
    double ttft_p95_ms;
    double tpot_p50_ms;
    double tpot_p95_ms;
    double latency_p50_ms;
    double latency_p95_ms;
    double peak_kv_blocks;
};

static int checked_int(std::size_t value, const char* message) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::overflow_error(message);
    }
    return static_cast<int>(value);
}

static std::vector<int> make_prompt(std::size_t length, int seed) {
    std::vector<int> tokens(length);
    tokens[0] = 50256;
    for (std::size_t i = 1; i < length; ++i) {
        tokens[i] =
            100 + (seed * 3571 + static_cast<int>(i) * 7919) % 50000;
    }
    return tokens;
}

static std::vector<WorkloadRequest> make_workload(
    std::size_t max_new_tokens) {
    const std::vector<std::size_t> prompt_lengths = {8, 16, 24, 32};
    std::vector<WorkloadRequest> workload;
    for (std::size_t i = 0; i < prompt_lengths.size(); ++i) {
        workload.push_back({
            i + 1,
            make_prompt(prompt_lengths[i], checked_int(i + 1, "seed overflow")),
            max_new_tokens,
        });
    }
    return workload;
}

static double elapsed_ms(
    Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

static int greedy_argmax(const float* logits, int vocab_size) {
    return static_cast<int>(
        std::max_element(logits, logits + vocab_size) - logits);
}

static RunMetrics finalize_metrics(
    std::string mode, int repeat, double total_seconds,
    std::size_t output_tokens, std::size_t peak_kv_blocks,
    std::vector<RequestMetrics> requests,
    std::vector<std::vector<int>> outputs) {
    std::vector<double> ttft;
    std::vector<double> tpot;
    std::vector<double> latency;
    for (const RequestMetrics& request : requests) {
        ttft.push_back(request.ttft_ms);
        tpot.push_back(request.tpot_ms);
        latency.push_back(request.latency_ms);
    }
    return {
        std::move(mode),
        repeat,
        total_seconds,
        output_tokens,
        output_tokens / total_seconds,
        percentile(ttft, 0.50),
        percentile(ttft, 0.95),
        percentile(tpot, 0.50),
        percentile(tpot, 0.95),
        percentile(latency, 0.50),
        percentile(latency, 0.95),
        peak_kv_blocks,
        std::move(requests),
        std::move(outputs),
    };
}

static RunMetrics run_full_recompute(
    const GPT2& model, const std::vector<WorkloadRequest>& workload,
    std::size_t max_context_length, int repeat) {
    GPT2DenseInferenceWorkspace workspace(
        model.config, /*max_batch_size=*/1,
        checked_int(max_context_length, "context is too large"));
    std::vector<RequestMetrics> request_metrics;
    std::vector<std::vector<int>> outputs;
    std::size_t output_token_count = 0;

    const Clock::time_point benchmark_start = Clock::now();
    for (const WorkloadRequest& request : workload) {
        std::vector<int> tokens = request.prompt_tokens;
        std::vector<int> generated;
        double first_token_ms = 0.0;
        for (std::size_t step = 0; step < request.max_new_tokens; ++step) {
            gpt2_forward_dense_with_workspace(
                &model, tokens.data(), 1, checked_int(
                    tokens.size(), "sequence is too long"), &workspace);
            const float* last_logits =
                workspace.acts().logits +
                (tokens.size() - 1) * model.config.padded_vocab_size;
            const int next_token =
                greedy_argmax(last_logits, model.config.vocab_size);
            tokens.push_back(next_token);
            generated.push_back(next_token);
            const double now_ms =
                elapsed_ms(benchmark_start, Clock::now());
            if (step == 0) first_token_ms = now_ms;
        }
        const double completion_ms =
            elapsed_ms(benchmark_start, Clock::now());
        const double tpot_ms =
            request.max_new_tokens > 1
                ? (completion_ms - first_token_ms) /
                      static_cast<double>(request.max_new_tokens - 1)
                : 0.0;
        request_metrics.push_back({
            request.request_id, first_token_ms, tpot_ms, completion_ms});
        output_token_count += generated.size();
        outputs.push_back(std::move(generated));
    }
    const double total_seconds =
        elapsed_ms(benchmark_start, Clock::now()) / 1000.0;
    return finalize_metrics(
        "full_recompute", repeat, total_seconds, output_token_count,
        /*peak_kv_blocks=*/0, std::move(request_metrics),
        std::move(outputs));
}

static RunMetrics run_paged_engine(
    GPT2& model, const std::vector<WorkloadRequest>& workload,
    std::size_t max_context_length, std::size_t max_num_sequences,
    const std::string& mode, int repeat) {
    std::size_t total_blocks = 0;
    for (const WorkloadRequest& request : workload) {
        const std::size_t max_processed_tokens =
            request.prompt_tokens.size() + request.max_new_tokens - 1;
        total_blocks +=
            (max_processed_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
    }
    GPT2Engine engine(
        model, total_blocks,
        {max_num_sequences, /*max_num_batched_tokens=*/64},
        max_context_length);

    std::vector<std::shared_ptr<Sequence>> sequences;
    for (const WorkloadRequest& request : workload) {
        sequences.push_back(engine.add_request(
            request.request_id, request.prompt_tokens,
            SamplingParams{request.max_new_tokens, -1, false}));
    }
    std::vector<Clock::time_point> first_token_times(workload.size());
    std::vector<Clock::time_point> completion_times(workload.size());
    std::vector<bool> saw_first_token(workload.size(), false);
    std::vector<bool> saw_completion(workload.size(), false);
    std::size_t peak_blocks = 0;

    const Clock::time_point benchmark_start = Clock::now();
    while (!engine.is_finished()) {
        std::vector<std::size_t> completion_counts;
        for (const auto& sequence : sequences) {
            completion_counts.push_back(
                sequence->num_completion_tokens());
        }
        const EngineStepResult step = engine.step();
        peak_blocks = std::max(
            peak_blocks,
            engine.num_blocks() - step.free_blocks_after_schedule);
        const Clock::time_point now = Clock::now();
        for (std::size_t i = 0; i < sequences.size(); ++i) {
            if (!saw_first_token[i] &&
                sequences[i]->num_completion_tokens() >
                    completion_counts[i]) {
                first_token_times[i] = now;
                saw_first_token[i] = true;
            }
            if (!saw_completion[i] && sequences[i]->is_finished()) {
                completion_times[i] = now;
                saw_completion[i] = true;
            }
        }
    }
    const Clock::time_point benchmark_end = Clock::now();

    std::vector<RequestMetrics> request_metrics;
    std::vector<std::vector<int>> outputs;
    std::size_t output_token_count = 0;
    for (std::size_t i = 0; i < workload.size(); ++i) {
        if (!saw_first_token[i] || !saw_completion[i]) {
            throw std::logic_error(
                "engine did not record complete request timings");
        }
        const double ttft_ms =
            elapsed_ms(benchmark_start, first_token_times[i]);
        const double completion_ms =
            elapsed_ms(benchmark_start, completion_times[i]);
        const double tpot_ms =
            workload[i].max_new_tokens > 1
                ? elapsed_ms(first_token_times[i], completion_times[i]) /
                      static_cast<double>(
                          workload[i].max_new_tokens - 1)
                : 0.0;
        request_metrics.push_back({
            workload[i].request_id, ttft_ms, tpot_ms, completion_ms});
        const Sequence& sequence = *sequences[i];
        outputs.emplace_back(
            sequence.token_ids().begin() +
                static_cast<std::ptrdiff_t>(
                    sequence.num_prompt_tokens()),
            sequence.token_ids().end());
        output_token_count += outputs.back().size();
    }

    return finalize_metrics(
        mode, repeat,
        elapsed_ms(benchmark_start, benchmark_end) / 1000.0,
        output_token_count, peak_blocks, std::move(request_metrics),
        std::move(outputs));
}

static void verify_same_outputs(
    const RunMetrics& expected, const RunMetrics& actual) {
    if (expected.outputs != actual.outputs) {
        throw std::runtime_error(
            "benchmark modes produced different greedy tokens");
    }
}

static ModeSummary summarize_mode(
    const std::vector<RunMetrics>& runs, const std::string& mode) {
    std::vector<double> total_seconds;
    std::vector<double> throughput;
    std::vector<double> ttft_p50;
    std::vector<double> ttft_p95;
    std::vector<double> tpot_p50;
    std::vector<double> tpot_p95;
    std::vector<double> latency_p50;
    std::vector<double> latency_p95;
    std::vector<double> peak_blocks;
    for (const RunMetrics& run : runs) {
        if (run.mode != mode) continue;
        total_seconds.push_back(run.total_seconds);
        throughput.push_back(run.output_tokens_per_second);
        ttft_p50.push_back(run.ttft_p50_ms);
        ttft_p95.push_back(run.ttft_p95_ms);
        tpot_p50.push_back(run.tpot_p50_ms);
        tpot_p95.push_back(run.tpot_p95_ms);
        latency_p50.push_back(run.latency_p50_ms);
        latency_p95.push_back(run.latency_p95_ms);
        peak_blocks.push_back(
            static_cast<double>(run.peak_kv_blocks));
    }
    if (total_seconds.empty()) {
        throw std::logic_error("cannot summarize a missing benchmark mode");
    }
    return {
        mode,
        percentile(total_seconds, 0.50),
        percentile(throughput, 0.50),
        percentile(ttft_p50, 0.50),
        percentile(ttft_p95, 0.50),
        percentile(tpot_p50, 0.50),
        percentile(tpot_p95, 0.50),
        percentile(latency_p50, 0.50),
        percentile(latency_p95, 0.50),
        percentile(peak_blocks, 0.50),
    };
}

static std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (char character : value) {
        switch (character) {
            case '\\': escaped << "\\\\"; break;
            case '"': escaped << "\\\""; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << character; break;
        }
    }
    return escaped.str();
}

static std::string cpu_model_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const std::string key = "model name";
        if (line.rfind(key, 0) == 0) {
            const std::size_t colon = line.find(':');
            return colon == std::string::npos
                       ? line
                       : line.substr(colon + 2);
        }
    }
    return "unknown";
}

static std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm time_info{};
    gmtime_r(&now, &time_info);
    char buffer[32];
    std::strftime(
        buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &time_info);
    return buffer;
}

static int openmp_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static void write_csv(
    const std::string& path, const std::vector<RunMetrics>& runs) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open CSV output");
    file << "mode,repeat,total_seconds,output_tokens,output_tokens_per_second,"
            "ttft_p50_ms,ttft_p95_ms,tpot_p50_ms,tpot_p95_ms,"
            "latency_p50_ms,latency_p95_ms,peak_kv_blocks\n";
    file << std::fixed << std::setprecision(6);
    for (const RunMetrics& run : runs) {
        file << run.mode << ',' << run.repeat << ','
             << run.total_seconds << ',' << run.output_tokens << ','
             << run.output_tokens_per_second << ','
             << run.ttft_p50_ms << ',' << run.ttft_p95_ms << ','
             << run.tpot_p50_ms << ',' << run.tpot_p95_ms << ','
             << run.latency_p50_ms << ',' << run.latency_p95_ms << ','
             << run.peak_kv_blocks << '\n';
    }
}

static void write_json(
    const std::string& path, const Options& options,
    const std::vector<WorkloadRequest>& workload,
    const std::vector<RunMetrics>& runs,
    const std::vector<ModeSummary>& summaries) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open JSON output");
    file << std::fixed << std::setprecision(6);
    file << "{\n"
         << "  \"metadata\": {\n"
         << "    \"timestamp_utc\": \"" << utc_timestamp() << "\",\n"
         << "    \"checkpoint\": \"" << json_escape(options.checkpoint)
         << "\",\n"
         << "    \"cpu\": \"" << json_escape(cpu_model_name()) << "\",\n"
         << "    \"hardware_threads\": "
         << std::thread::hardware_concurrency() << ",\n"
         << "    \"openmp_threads\": " << openmp_threads() << ",\n"
         << "    \"compiler\": \"" << json_escape(__VERSION__) << "\",\n"
         << "    \"git_commit\": \"" << MINI_VLLM_GIT_COMMIT << "\",\n"
         << "    \"build_flags\": \""
         << json_escape(MINI_VLLM_BUILD_FLAGS) << "\",\n"
         << "    \"arrival_policy\": \"all_requests_at_time_zero\",\n"
         << "    \"timing_clock\": \"steady_clock\",\n"
         << "    \"scheduler_token_budget\": 64,\n"
         << "    \"kv_block_size\": " << PAGE_SIZE << ",\n"
         << "    \"warmup_runs_per_mode\": 1,\n"
         << "    \"timed_repeats\": " << options.repeats << "\n"
         << "  },\n"
         << "  \"workload\": [\n";
    for (std::size_t i = 0; i < workload.size(); ++i) {
        file << "    {\"request_id\": " << workload[i].request_id
             << ", \"prompt_tokens\": "
             << workload[i].prompt_tokens.size()
             << ", \"output_tokens\": "
             << workload[i].max_new_tokens
             << ", \"prompt_token_ids\": [";
        for (std::size_t token_index = 0;
             token_index < workload[i].prompt_tokens.size();
             ++token_index) {
            if (token_index != 0) file << ", ";
            file << workload[i].prompt_tokens[token_index];
        }
        file << "]}"
             << (i + 1 == workload.size() ? "\n" : ",\n");
    }
    file << "  ],\n  \"runs\": [\n";
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const RunMetrics& run = runs[i];
        file << "    {\n"
             << "      \"mode\": \"" << run.mode << "\",\n"
             << "      \"repeat\": " << run.repeat << ",\n"
             << "      \"total_seconds\": " << run.total_seconds << ",\n"
             << "      \"output_tokens\": " << run.output_tokens << ",\n"
             << "      \"output_tokens_per_second\": "
             << run.output_tokens_per_second << ",\n"
             << "      \"ttft_p50_ms\": " << run.ttft_p50_ms << ",\n"
             << "      \"ttft_p95_ms\": " << run.ttft_p95_ms << ",\n"
             << "      \"tpot_p50_ms\": " << run.tpot_p50_ms << ",\n"
             << "      \"tpot_p95_ms\": " << run.tpot_p95_ms << ",\n"
             << "      \"latency_p50_ms\": "
             << run.latency_p50_ms << ",\n"
             << "      \"latency_p95_ms\": "
             << run.latency_p95_ms << ",\n"
             << "      \"peak_kv_blocks\": "
             << run.peak_kv_blocks << ",\n"
             << "      \"generated_token_ids\": [";
        for (std::size_t request_index = 0;
             request_index < run.outputs.size(); ++request_index) {
            if (request_index != 0) file << ", ";
            file << '[';
            for (std::size_t token_index = 0;
                 token_index < run.outputs[request_index].size();
                 ++token_index) {
                if (token_index != 0) file << ", ";
                file << run.outputs[request_index][token_index];
            }
            file << ']';
        }
        file << "],\n"
             << "      \"requests\": [\n";
        for (std::size_t j = 0; j < run.requests.size(); ++j) {
            const RequestMetrics& request = run.requests[j];
            file << "        {\"request_id\": " << request.request_id
                 << ", \"ttft_ms\": " << request.ttft_ms
                 << ", \"tpot_ms\": " << request.tpot_ms
                 << ", \"latency_ms\": " << request.latency_ms << "}"
                 << (j + 1 == run.requests.size() ? "\n" : ",\n");
        }
        file << "      ]\n    }"
             << (i + 1 == runs.size() ? "\n" : ",\n");
    }
    file << "  ],\n  \"summary_median_across_repeats\": [\n";
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const ModeSummary& summary = summaries[i];
        file << "    {\"mode\": \"" << summary.mode
             << "\", \"total_seconds\": " << summary.total_seconds
             << ", \"output_tokens_per_second\": "
             << summary.output_tokens_per_second
             << ", \"ttft_p50_ms\": " << summary.ttft_p50_ms
             << ", \"ttft_p95_ms\": " << summary.ttft_p95_ms
             << ", \"tpot_p50_ms\": " << summary.tpot_p50_ms
             << ", \"tpot_p95_ms\": " << summary.tpot_p95_ms
             << ", \"latency_p50_ms\": "
             << summary.latency_p50_ms
             << ", \"latency_p95_ms\": "
             << summary.latency_p95_ms
             << ", \"peak_kv_blocks\": "
             << summary.peak_kv_blocks << "}"
             << (i + 1 == summaries.size() ? "\n" : ",\n");
    }
    file << "  ]\n}\n";
}

static void print_run(const RunMetrics& run) {
    std::cout << std::fixed << std::setprecision(3)
              << run.mode << " repeat=" << run.repeat
              << " total=" << run.total_seconds << " s"
              << " throughput=" << run.output_tokens_per_second
              << " tok/s"
              << " TTFT(P50/P95)=" << run.ttft_p50_ms << '/'
              << run.ttft_p95_ms << " ms"
              << " TPOT(P50/P95)=" << run.tpot_p50_ms << '/'
              << run.tpot_p95_ms << " ms"
              << " latency(P50/P95)=" << run.latency_p50_ms << '/'
              << run.latency_p95_ms << " ms"
              << " peak_blocks=" << run.peak_kv_blocks << '\n';
}

static void print_summary(const ModeSummary& summary) {
    std::cout << std::fixed << std::setprecision(3)
              << "中位数 " << summary.mode
              << " total=" << summary.total_seconds << " s"
              << " throughput=" << summary.output_tokens_per_second
              << " tok/s"
              << " TTFT(P50/P95)=" << summary.ttft_p50_ms << '/'
              << summary.ttft_p95_ms << " ms"
              << " TPOT(P50/P95)=" << summary.tpot_p50_ms << '/'
              << summary.tpot_p95_ms << " ms"
              << " latency(P50/P95)=" << summary.latency_p50_ms << '/'
              << summary.latency_p95_ms << " ms"
              << " peak_blocks=" << summary.peak_kv_blocks << '\n';
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (argument == "--checkpoint") {
            options.checkpoint = require_value("--checkpoint");
        } else if (argument == "--json") {
            options.json_path = require_value("--json");
        } else if (argument == "--csv") {
            options.csv_path = require_value("--csv");
        } else if (argument == "--repeats") {
            options.repeats = std::stoi(require_value("--repeats"));
        } else if (argument == "--max-new-tokens") {
            options.max_new_tokens = static_cast<std::size_t>(
                std::stoul(require_value("--max-new-tokens")));
        } else {
            throw std::invalid_argument(
                "unknown benchmark option: " + argument);
        }
    }
    if (options.repeats <= 0 || options.max_new_tokens < 2) {
        throw std::invalid_argument(
            "repeats must be positive and max-new-tokens must be at least 2");
    }
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<WorkloadRequest> workload =
            make_workload(options.max_new_tokens);
        std::size_t max_context_length = 0;
        for (const WorkloadRequest& request : workload) {
            max_context_length = std::max(
                max_context_length,
                request.prompt_tokens.size() +
                    request.max_new_tokens - 1);
        }

        GPT2 model{};
        gpt2_build_from_checkpoint(
            &model, options.checkpoint.c_str());

        std::cout << "预热三种执行模式（不计时）...\n";
        const RunMetrics warmup_full =
            run_full_recompute(model, workload, max_context_length, -1);
        const RunMetrics warmup_sequential =
            run_paged_engine(
                model, workload, max_context_length,
                /*max_num_sequences=*/1, "paged_sequential", -1);
        const RunMetrics warmup_continuous =
            run_paged_engine(
                model, workload, max_context_length, workload.size(),
                "continuous_batching", -1);
        verify_same_outputs(warmup_full, warmup_sequential);
        verify_same_outputs(warmup_full, warmup_continuous);

        std::vector<RunMetrics> runs;
        for (int repeat = 1; repeat <= options.repeats; ++repeat) {
            RunMetrics full =
                run_full_recompute(
                    model, workload, max_context_length, repeat);
            RunMetrics sequential =
                run_paged_engine(
                    model, workload, max_context_length,
                    /*max_num_sequences=*/1, "paged_sequential", repeat);
            RunMetrics continuous =
                run_paged_engine(
                    model, workload, max_context_length, workload.size(),
                    "continuous_batching", repeat);
            verify_same_outputs(full, sequential);
            verify_same_outputs(full, continuous);
            print_run(full);
            print_run(sequential);
            print_run(continuous);
            runs.push_back(std::move(full));
            runs.push_back(std::move(sequential));
            runs.push_back(std::move(continuous));
        }

        const std::vector<ModeSummary> summaries = {
            summarize_mode(runs, "full_recompute"),
            summarize_mode(runs, "paged_sequential"),
            summarize_mode(runs, "continuous_batching"),
        };
        std::cout << "正式结果中位数：\n";
        for (const ModeSummary& summary : summaries) {
            print_summary(summary);
        }

        const std::filesystem::path json_parent =
            std::filesystem::path(options.json_path).parent_path();
        const std::filesystem::path csv_parent =
            std::filesystem::path(options.csv_path).parent_path();
        if (!json_parent.empty()) {
            std::filesystem::create_directories(json_parent);
        }
        if (!csv_parent.empty()) {
            std::filesystem::create_directories(csv_parent);
        }
        write_json(
            options.json_path, options, workload, runs, summaries);
        write_csv(options.csv_path, runs);
        std::cout << "原始结果已写入 " << options.json_path
                  << " 和 " << options.csv_path << '\n';

        gpt2_free(&model);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark 失败：" << error.what() << '\n';
        return 1;
    }
}
