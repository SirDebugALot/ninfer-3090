#include "ninfer/ops/gdn_input_proj.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 GDN-input-projection Op.
constexpr ReductionCriterion kGdnInputProjA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kGdnInputProjA4Tolerance{0.16, 4.0e-3, 0.16};

constexpr std::int32_t kQ4Q5Hidden        = 5120;
constexpr std::int32_t kQ4Q5QkRows        = 4096;
constexpr std::int32_t kQ4Q5ValueRows     = 6144;
constexpr std::int32_t kQ4Q5ValueZRows    = 12288;
constexpr std::int32_t kQ4Q5QkvRows       = kQ4Q5QkRows + kQ4Q5ValueRows;
constexpr std::int32_t kQ4Q5AllParentRows = kQ4Q5QkRows + kQ4Q5ValueZRows;
constexpr std::size_t kGdnBlasScratchBytes = 16384;

int run_q4_q5_resource_cases(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight);

int verify_output_range(std::string_view label, const GuardedBf16Tensor& output,
                        std::int32_t full_rows, std::int32_t output_row_offset,
                        std::int32_t output_rows, const quantized_weight::PackedWeight& weight,
                        std::int32_t weight_row_offset, const std::vector<float>& activation,
                        std::int32_t hidden, std::int32_t tokens) {
    const std::vector<double> actual =
        gather_rows(output.values(), full_rows, output_row_offset, output_rows, tokens);
    const std::vector<double> expected =
        projection_oracle(weight, weight_row_offset, output_rows, activation, hidden, tokens);
    return compare(label, actual, expected, kGdnInputProjA16Tolerance);
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkRows      = 4096;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkRows + kValueRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 401U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output   = qkv.tensor();
    Tensor z_output = z.tensor();
    ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q4/Q5 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qk" + suffix, qkv, kRows, 0, kQkRows, query_key.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn value" + suffix, qkv, kRows, kQkRows, kValueRows,
                                    value_z_weight.host, 0, activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, value_z_weight.host,
                                    kValueRows, activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

int run_q4_q5() {
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, kQ4Q5QkRows, kQ4Q5Hidden,
                                                409U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, kQ4Q5ValueZRows, kQ4Q5Hidden,
                                                419U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 16, 17}) {
        failures += run_q4_q5_case(query_key, value_z_weight, tokens);
    }
    failures += run_q4_q5_resource_cases(query_key, value_z_weight);
    return failures;
}

int run_w8_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQkvRows     = 8192;
    constexpr std::int32_t kZRows       = 4096;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 501U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qkv" + suffix, qkv, kQkvRows, 0, kQkvRows, parent.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 503U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 97}) { failures += run_w8_case(parent, tokens); }
    return failures;
}

int verify_output_range_sampled_bits(std::string_view label,
                                     std::span<const std::uint16_t> output_bits,
                                     std::int32_t full_rows,
                                     std::int32_t output_row_offset,
                                     std::int32_t output_rows,
                                     const quantized_weight::PackedWeight& weight,
                                     std::int32_t weight_row_offset,
                                     const std::vector<float>& activation, std::int32_t hidden,
                                     std::int32_t tokens, const ReductionCriterion& criterion) {
    const std::vector<std::int32_t> rows = sampled_rows(output_rows);
    std::vector<std::int32_t> selected_tokens;
    for (const std::int32_t token :
         {0, 1, tokens / 4, tokens / 2, (3 * tokens) / 4, tokens - 2, tokens - 1}) {
        if (token >= 0 && token < tokens &&
            std::find(selected_tokens.begin(), selected_tokens.end(), token) ==
                selected_tokens.end()) {
            selected_tokens.push_back(token);
        }
    }
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * selected_tokens.size());
    expected.reserve(rows.size() * selected_tokens.size());
    for (const std::int32_t local_row : rows) {
        const std::int32_t output_row = output_row_offset + local_row;
        const std::int32_t weight_row = weight_row_offset + local_row;
        for (const std::int32_t token : selected_tokens) {
            actual.push_back(bf16_to_f32(
                output_bits[static_cast<std::size_t>(token) * full_rows + output_row]));
            expected.push_back(quantized_weight::dot_fp64(
                weight, weight_row, activation.data() + static_cast<std::size_t>(token) * hidden,
                hidden));
        }
    }
    return compare(label, actual, expected, criterion);
}

int verify_output_range_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                std::int32_t full_rows, std::int32_t output_row_offset,
                                std::int32_t output_rows,
                                const quantized_weight::PackedWeight& weight,
                                std::int32_t weight_row_offset,
                                const std::vector<float>& activation, std::int32_t hidden,
                                std::int32_t tokens, const ReductionCriterion& criterion) {
    const std::vector<std::uint16_t> bits = output.bits();
    return verify_output_range_sampled_bits(label, bits, full_rows, output_row_offset, output_rows,
                                            weight, weight_row_offset, activation, hidden, tokens,
                                            criterion);
}

std::size_t expected_q4_q5_workspace(std::int32_t tokens) {
    if (tokens < 1024 || tokens % 128 != 0) { return 0; }
    const std::size_t parent_bytes = static_cast<std::size_t>(kQ4Q5AllParentRows) *
                                     static_cast<std::size_t>(kQ4Q5Hidden) * sizeof(std::uint16_t);
    return parent_bytes + kGdnBlasScratchBytes;
}

int verify_q4_q5_capacity(std::string_view label, std::int32_t min_tokens,
                          std::int32_t max_tokens, std::size_t expected) {
    const std::size_t actual = ops::gdn_input_proj_workspace_capacity_bytes(
        kQ4Q5Hidden, kQ4Q5QkRows, kQ4Q5ValueZRows, min_tokens, max_tokens);
    if (actual == expected) { return 0; }
    std::cerr << label << ": workspace capacity got=" << actual << " expected=" << expected
              << '\n';
    return 1;
}

int verify_finite_bits(std::string_view label, std::span<const std::uint16_t> bits) {
    for (std::size_t index = 0; index < bits.size(); ++index) {
        if (!std::isfinite(bf16_to_f32(bits[index]))) {
            std::cerr << label << ": output element " << index << " was not finite\n";
            return 1;
        }
    }
    return 0;
}

void poison_q4_q5_outputs(GuardedBf16Tensor& qkv, GuardedBf16Tensor& z,
                           std::int32_t tokens) {
    cuda_check(cudaMemset(qkv.data(), 0xff, static_cast<std::size_t>(kQ4Q5QkvRows) * tokens *
                                                sizeof(std::uint16_t)),
               "poison Q4/Q5 qkv output");
    cuda_check(cudaMemset(z.data(), 0xff, static_cast<std::size_t>(kQ4Q5ValueRows) * tokens *
                                              sizeof(std::uint16_t)),
               "poison Q4/Q5 z output");
}

int verify_q4_q5_resource_output(std::string_view label, const GuardedBf16Tensor& qkv,
                                  const GuardedBf16Tensor& z,
                                  const DevicePackedWeight& query_key,
                                  const DevicePackedWeight& value_z_weight,
                                  const std::vector<float>& activation, std::int32_t tokens) {
    const std::vector<std::uint16_t> qkv_bits = qkv.bits();
    const std::vector<std::uint16_t> z_bits   = z.bits();
    int failures = qkv.verify_guards(std::string(label) + " qkv");
    failures += z.verify_guards(std::string(label) + " z");
    failures += verify_finite_bits(std::string(label) + " qkv", qkv_bits);
    failures += verify_finite_bits(std::string(label) + " z", z_bits);
    failures += verify_output_range_sampled_bits(
        std::string(label) + " query", qkv_bits, kQ4Q5QkvRows, 0, 2048, query_key.host, 0,
        activation, kQ4Q5Hidden, tokens, kGdnInputProjA16Tolerance);
    failures += verify_output_range_sampled_bits(
        std::string(label) + " key", qkv_bits, kQ4Q5QkvRows, 2048, 2048, query_key.host, 2048,
        activation, kQ4Q5Hidden, tokens, kGdnInputProjA16Tolerance);
    failures += verify_output_range_sampled_bits(
        std::string(label) + " value", qkv_bits, kQ4Q5QkvRows, kQ4Q5QkRows, kQ4Q5ValueRows,
        value_z_weight.host, 0, activation, kQ4Q5Hidden, tokens, kGdnInputProjA16Tolerance);
    failures += verify_output_range_sampled_bits(
        std::string(label) + " z", z_bits, kQ4Q5ValueRows, 0, kQ4Q5ValueRows,
        value_z_weight.host, kQ4Q5ValueRows, activation, kQ4Q5Hidden, tokens,
        kGdnInputProjA16Tolerance);
    return failures;
}

int run_q4_q5_resource_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                             DeviceContext& context, std::int32_t tokens, bool provide_blas,
                             bool graph_replay) {
    const std::vector<float> activation =
        make_bf16_activation(kQ4Q5Hidden, tokens, 431U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQ4Q5QkvRows, tokens);
    GuardedBf16Tensor z(kQ4Q5ValueRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kQ4Q5Hidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();

    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        kQ4Q5Hidden, kQ4Q5QkRows, kQ4Q5ValueZRows, tokens, tokens);
    const std::size_t expected_peak = provide_blas ? expected_q4_q5_workspace(tokens) : 0;
    GuardedDeviceBuffer workspace_storage(std::max<std::size_t>(capacity, 256));
    workspace_storage.fill(0x3c);
    WorkspaceArena workspace(
        DeviceSpan{workspace_storage.data(), workspace_storage.bytes()});
    const cublasHandle_t blas = provide_blas ? context.blas : nullptr;
    const std::string label   = "gdn Q4/Q5 resource T=" + std::to_string(tokens) +
                              (provide_blas ? " cuBLAS" : " null-handle");

    // Uploads and poison fills use the default stream; the execution stream is nonblocking.
    cuda_synchronize();
    workspace.reset();
    workspace.reset_peak();
    ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), qkv_output, z_output,
                        workspace, context.stream, blas);
    cuda_synchronize(context.stream);

    int failures = 0;
    if (workspace.used() != 0 || workspace.peak_used() != expected_peak) {
        std::cerr << label << ": workspace high-water got=" << workspace.peak_used()
                  << " expected=" << expected_peak << " used=" << workspace.used() << '\n';
        ++failures;
    }
    failures += workspace_storage.verify_guards(label + " workspace");
    failures += verify_q4_q5_resource_output(label, qkv, z, query_key, value_z_weight, activation,
                                              tokens);
    failures += verify_preserved(label + " x", device_activation, activation_bits);
    failures += query_key.verify_preserved(label + " query/key weight");
    failures += value_z_weight.verify_preserved(label + " value/z weight");

    if (graph_replay) {
        poison_q4_q5_outputs(qkv, z, tokens);
        cuda_synchronize();
        workspace.reset();
        workspace.reset_peak();
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(context.stream, [&] {
            ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), qkv_output, z_output,
                                workspace, context.stream, blas);
        });
        executable.instantiate(definition);
        if (workspace.used() != 0 || workspace.peak_used() != expected_peak) {
            std::cerr << label << ": graph capture workspace high-water got="
                      << workspace.peak_used() << " expected=" << expected_peak
                      << " used=" << workspace.used() << '\n';
            ++failures;
        }
        failures += workspace_storage.verify_guards(label + " graph capture workspace");
        for (int replay = 0; replay < 2; ++replay) {
            poison_q4_q5_outputs(qkv, z, tokens);
            cuda_synchronize();
            executable.launch(context.stream);
            cuda_synchronize(context.stream);
            const std::string replay_label =
                label + " graph replay=" + std::to_string(replay);
            failures += verify_q4_q5_resource_output(replay_label, qkv, z, query_key,
                                                      value_z_weight, activation, tokens);
            failures += workspace_storage.verify_guards(replay_label + " workspace");
            failures += verify_preserved(replay_label + " x", device_activation,
                                         activation_bits);
        }
        failures += query_key.verify_preserved(label + " graph query/key weight");
        failures += value_z_weight.verify_preserved(label + " graph value/z weight");
    }
    return failures;
}

int run_q4_q5_resource_cases(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight) {
    constexpr std::array<std::int32_t, 6> kTokens{1023, 1024, 1025, 1151, 1152, 2048};
    int failures = 0;
    for (const std::int32_t tokens : kTokens) {
        failures += verify_q4_q5_capacity("gdn Q4/Q5 exact T=" + std::to_string(tokens), tokens,
                                          tokens, expected_q4_q5_workspace(tokens));
    }
    failures += verify_q4_q5_capacity("gdn Q4/Q5 interval 1000..1100", 1000, 1100,
                                      expected_q4_q5_workspace(1024));
    failures += verify_q4_q5_capacity("gdn Q4/Q5 interval 1025..1025", 1025, 1025, 0);
    failures += verify_q4_q5_capacity("gdn Q4/Q5 interval 1025..1152", 1025, 1152,
                                      expected_q4_q5_workspace(1152));

    DeviceContext context;
    for (const std::int32_t tokens : kTokens) {
        failures += run_q4_q5_resource_case(query_key, value_z_weight, context, tokens, true,
                                             tokens == 1024);
    }
    failures +=
        run_q4_q5_resource_case(query_key, value_z_weight, context, 1024, false, false);
    return failures;
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkvRows     = 10240;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const bool a4                       = policy == ops::LinearPolicy::AllowA4;
    const ReductionCriterion& criterion = a4 ? kGdnInputProjA4Tolerance : kGdnInputProjA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens);
    int failures = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048,
                                            parent.host, 0, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, kHidden, tokens, criterion);
    failures += verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host,
                                            kQkvRows, activation, kHidden, tokens, criterion);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 607U, options));
    int failures = 0;
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only);
    failures += run_nvfp4_case(parent, 4, ops::LinearPolicy::A16Only);
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 2, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 17, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 1024, ops::LinearPolicy::AllowA4);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    bool q4_q5_only = false;
    if (argc == 2 && std::string_view(argv[1]) == "--q4-q5-only") {
        q4_q5_only = true;
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [--q4-q5-only]\n";
        return 2;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    if (!q4_q5_only) {
        failures += run_w8();
        failures += run_nvfp4();
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
