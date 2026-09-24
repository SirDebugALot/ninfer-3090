// Public-Op benchmark for the Qwen3.6/Qwen3.8 prompt GQA path.
//
// The benchmark enters through either gqa_attention_cached() or the model's append-plus-
// attention gqa_attention() API, exercising their distinct metadata specializations. It can
// model ordinary INT8 K/V or the target's rotated-K / packed-V RK8V4 cache and can either
// retain the working set or evict L2 before every measured launch.

#include "ninfer/ops/gqa_attention.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQHeads     = 24;
constexpr std::int32_t kKvHeads    = 4;
constexpr std::int32_t kQuantGroup = 64;
constexpr float kScale             = 0.0625F;
constexpr std::size_t kFlushBytes  = std::size_t{64} << 20;

enum class KvMode : std::uint8_t { Int8, Rk8v4 };
enum class Execution : std::uint8_t { Eager, Graph };
enum class CacheState : std::uint8_t { Warm, Cold };
enum class Route : std::uint8_t { Cached, Append };

struct Options {
    KvMode kv_mode          = KvMode::Rk8v4;
    Execution execution     = Execution::Graph;
    CacheState cache_state  = CacheState::Warm;
    Route route             = Route::Cached;
    std::int32_t tokens     = 2048;
    std::int32_t base       = 0;
    int warmup              = 5;
    int repeat              = 25;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_gqa_attention_bench [--kv int8|rk8v4] "
                 "[--route cached|append] "
                 "[--tokens T] [--base N] [--execution eager|graph] "
                 "[--cache warm|cold] [--warmup N] [--repeat N]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--kv") {
            const std::string_view value(next("--kv requires a value"));
            if (value == "int8")
                options.kv_mode = KvMode::Int8;
            else if (value == "rk8v4")
                options.kv_mode = KvMode::Rk8v4;
            else
                usage("--kv expects int8 or rk8v4");
        } else if (argument == "--route") {
            const std::string_view value(next("--route requires a value"));
            if (value == "cached")
                options.route = Route::Cached;
            else if (value == "append")
                options.route = Route::Append;
            else
                usage("--route expects cached or append");
        } else if (argument == "--tokens") {
            options.tokens = parse_i32(next("--tokens requires a value"), 17, 8192, "--tokens");
        } else if (argument == "--base") {
            options.base = parse_i32(next("--base requires a value"), 0, 262143, "--base");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else
                usage("--execution expects eager or graph");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "warm")
                options.cache_state = CacheState::Warm;
            else if (value == "cold")
                options.cache_state = CacheState::Cold;
            else
                usage("--cache expects warm or cold");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    if (options.base > std::numeric_limits<std::int32_t>::max() - options.tokens ||
        static_cast<std::uint32_t>(options.base + options.tokens) >
            ops::kGqaAttentionMaximumVisibleKeys) {
        usage("base + tokens exceeds the public GQA context limit");
    }
    return options;
}

std::int32_t align_pages(std::int32_t tokens) {
    return ((tokens + kPagedKVPageSize - 1) / kPagedKVPageSize) * kPagedKVPageSize;
}

class GqaCase {
public:
    explicit GqaCase(const Options& options)
        : options_(options), visible_(options.base + options.tokens), capacity_(align_pages(visible_)),
          pages_(capacity_ / kPagedKVPageSize), packed_v_(options.kv_mode == KvMode::Rk8v4),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQHeads * options.tokens)),
          positions_(static_cast<std::size_t>(options.tokens) * sizeof(std::int32_t)),
          out_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kQHeads * options.tokens * 2)),
          cache_k_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                                     kKvHeads * pages_)),
          cache_v_(bench::make_zeros(static_cast<std::size_t>(packed_v_ ? kHeadDim / 2 : kHeadDim) *
                                     kPagedKVPageSize * kKvHeads * pages_)),
          cache_k_scale_(bench::make_zeros(scale_bytes())),
          cache_v_scale_(bench::make_zeros(scale_bytes())),
          block_table_(static_cast<std::size_t>(pages_) * sizeof(std::int32_t)),
          workspace_(256), q_tensor_(q_.p, DType::BF16, {kHeadDim, kQHeads, options.tokens}),
          positions_tensor_(positions_.p, DType::I32, {options.tokens}),
          out_tensor_(out_.p, DType::BF16, {kHeadDim, kQHeads, options.tokens}) {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(options.tokens));
        for (std::int32_t token = 0; token < options.tokens; ++token) {
            host_positions[static_cast<std::size_t>(token)] = options.base + token;
        }
        std::vector<std::int32_t> host_table(static_cast<std::size_t>(pages_));
        for (std::int32_t page = 0; page < pages_; ++page) {
            host_table[static_cast<std::size_t>(page)] = page;
        }
        const std::size_t scale_elements = scale_bytes() / sizeof(std::uint16_t);
        std::vector<std::uint16_t> host_scales(scale_elements, 0x2000u);

        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(block_table_.p, host_table.data(), block_table_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cache_k_scale_.p, host_scales.data(), cache_k_scale_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cache_v_scale_.p, host_scales.data(), cache_v_scale_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(cache_k_.p, 0x11, cache_k_.bytes));
        CUDA_CHECK(cudaMemset(cache_v_.p, packed_v_ ? 0x87 : 0x21, cache_v_.bytes));

        cache_view_ = {
            .k_pages = Tensor(cache_k_.p, DType::I8,
                              {kHeadDim, kPagedKVPageSize, kKvHeads, pages_}),
            .v_pages = Tensor(cache_v_.p, packed_v_ ? DType::U8 : DType::I8,
                              {packed_v_ ? kHeadDim / 2 : kHeadDim, kPagedKVPageSize, kKvHeads,
                               pages_}),
            .k_scale_pages = Tensor(cache_k_scale_.p, DType::FP16,
                                    {kHeadDim / kQuantGroup, kPagedKVPageSize, kKvHeads, pages_}),
            .v_scale_pages = Tensor(cache_v_scale_.p, DType::FP16,
                                    {kHeadDim / kQuantGroup, kPagedKVPageSize, kKvHeads, pages_}),
            .block_table  = Tensor(block_table_.p, DType::I32, {pages_}),
            .head_dim     = kHeadDim,
            .num_kv_heads = kKvHeads,
            .dtype        = DType::I8,
            .quant_group  = kQuantGroup,
            .packed_v     = packed_v_,
            .rotate_k     = packed_v_,
            .rotate_v     = packed_v_,
        };
        if (options.route == Route::Append) {
            const std::size_t kv_elements =
                static_cast<std::size_t>(kHeadDim) * kKvHeads * options.tokens;
            k_ = bench::make_bf16(kv_elements);
            v_ = bench::make_bf16(kv_elements);
            table_row_ = bench::make_zeros(sizeof(std::int32_t));
            k_tensor_ = Tensor(k_.p, DType::BF16, {kHeadDim, kKvHeads, options.tokens});
            v_tensor_ = Tensor(v_.p, DType::BF16, {kHeadDim, kKvHeads, options.tokens});
            table_row_tensor_ = Tensor(table_row_.p, DType::I32, {1});
            batch_cache_view_ = {
                .k_pages = cache_view_.k_pages,
                .v_pages = cache_view_.v_pages,
                .k_scale_pages = cache_view_.k_scale_pages,
                .v_scale_pages = cache_view_.v_scale_pages,
                .block_tables = Tensor(block_table_.p, DType::I32, {pages_, 1}),
                .head_dim = kHeadDim,
                .num_kv_heads = kKvHeads,
                .dtype = DType::I8,
                .quant_group = kQuantGroup,
                .packed_v = packed_v_,
                .rotate_k = packed_v_,
                .rotate_v = packed_v_,
            };
        }
    }

    void launch(cudaStream_t stream) {
        WorkspaceArena workspace(DeviceSpan{workspace_.p, workspace_.bytes});
        const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(visible_),
                                                  static_cast<std::uint32_t>(capacity_)};
        if (options_.route == Route::Append) {
            // One active request and no valid-columns mask select the same metadata
            // specialization as model prefill. Replays overwrite the same append interval.
            ops::gqa_attention(q_tensor_, k_tensor_, v_tensor_, positions_tensor_, Tensor{},
                               table_row_tensor_, kScale, batch_cache_view_, envelope,
                               workspace, out_tensor_, stream);
        } else {
            ops::gqa_attention_cached(q_tensor_, positions_tensor_, kScale, cache_view_, envelope,
                                      workspace, out_tensor_, stream);
        }
    }

private:
    [[nodiscard]] std::size_t scale_bytes() const {
        return static_cast<std::size_t>(kHeadDim / kQuantGroup) * kPagedKVPageSize * kKvHeads *
               pages_ * sizeof(std::uint16_t);
    }

    Options options_;
    std::int32_t visible_;
    std::int32_t capacity_;
    std::int32_t pages_;
    bool packed_v_;
    DeviceBuffer q_;
    DeviceBuffer positions_;
    DeviceBuffer out_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer block_table_;
    DeviceBuffer workspace_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer table_row_;
    Tensor q_tensor_;
    Tensor positions_tensor_;
    Tensor out_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor table_row_tensor_;
    PagedKVLayerView cache_view_;
    PagedKVBatchLayerView batch_cache_view_;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        GqaCase test_case(options);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::ColdTiming timing;

        if (options.execution == Execution::Graph) {
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t capture_stream) { test_case.launch(capture_stream); });
            if (options.cache_state == CacheState::Cold) {
                DeviceBuffer flush(kFlushBytes);
                timing = bench::measure_cold_graph(graph, flush, stream, options.warmup,
                                                   options.repeat);
            } else {
                timing = bench::measure_graph(graph, stream, options.warmup, options.repeat);
            }
        } else {
            const auto launch = [&](cudaStream_t launch_stream) { test_case.launch(launch_stream); };
            if (options.cache_state == CacheState::Cold) {
                DeviceBuffer flush(kFlushBytes);
                timing = bench::measure_cold_launch(launch, flush, stream, options.warmup,
                                                    options.repeat);
            } else {
                timing = bench::measure_launch(launch, stream, options.warmup, options.repeat);
            }
        }

        const double token_rate = static_cast<double>(options.tokens) * 1.0e6 / timing.median_us;
        std::printf("route=%s api=%s kv=%s execution=%s cache=%s tokens=%d base=%d median_us=%.3f "
                    "min_us=%.3f p95_us=%.3f query_tok_s=%.2f\n",
                    options.route == Route::Append ? "append" : "cached",
                    options.route == Route::Append ? "gqa_attention" : "gqa_attention_cached",
                    options.kv_mode == KvMode::Rk8v4 ? "rk8v4" : "int8",
                    options.execution == Execution::Graph ? "graph" : "eager",
                    options.cache_state == CacheState::Warm ? "warm" : "cold", options.tokens,
                    options.base, timing.median_us, timing.min_us, timing.p95_us, token_rate);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gqa_attention_bench: %s\n", error.what());
        return 1;
    }
}
