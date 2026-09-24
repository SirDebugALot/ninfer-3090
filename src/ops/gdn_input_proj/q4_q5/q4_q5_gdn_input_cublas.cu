#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden = 5120;
constexpr int kQkRows = 4096;
constexpr int kAllRows = 16384;
constexpr std::size_t kQ4Groups = std::size_t(kQkRows) * kHidden / 64;
constexpr std::size_t kAllGroups = std::size_t(kAllRows) * kHidden / 64;
constexpr unsigned kDequantThreads       = 256;
constexpr unsigned kDequantSubgroupWidth = 8;
constexpr unsigned kDequantGroupsPerWarp = 4;
constexpr unsigned kDequantGroupsPerBlock =
    (kDequantThreads / 32) * kDequantGroupsPerWarp;

union alignas(16) DecodedBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

static_assert(sizeof(DecodedBf16x8) == sizeof(uint4));

void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Q4/Q5 GDN input cuBLAS status " + std::to_string(int(status)));
    }
}

__global__ void dequantize_parents(const std::uint8_t* q4_codes, const std::uint16_t* q4_scales,
                                   const std::uint8_t* q5_codes, const std::uint8_t* q5_high,
                                   const std::uint16_t* q5_scales, __nv_bfloat162* weights) {
    const unsigned lane          = threadIdx.x & 31U;
    const unsigned subgroup      = lane / kDequantSubgroupWidth;
    const unsigned subgroup_lane = lane % kDequantSubgroupWidth;
    const std::size_t warp =
        (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::size_t output_group = warp * kDequantGroupsPerWarp + subgroup;
    if (output_group >= kAllGroups) { return; }

    // The concatenated decoded matrix contains all Q4 groups first and all Q5
    // groups second. Both boundaries are multiples of a complete 32-group CTA,
    // so each eight-lane subgroup, warp, and CTA follows one uniform format.
    const bool five = output_group >= kQ4Groups;
    const std::size_t group = five ? output_group - kQ4Groups : output_group;

    float scale = 0.0f;
    if (subgroup_lane == 0) {
        scale = __half2float(
            __ushort_as_half((five ? q5_scales : q4_scales)[group]));
    }
    const unsigned active = __activemask();
    scale = __shfl_sync(active, scale, int(subgroup * kDequantSubgroupWidth));

    const auto* codes = five ? q5_codes : q4_codes;
    const auto* code_ptr = codes + group * 32 + subgroup_lane * 4;
    const unsigned packed = *reinterpret_cast<const std::uint32_t*>(code_ptr);
    const unsigned upper = five ? q5_high[group * 8 + subgroup_lane] : 0U;
    DecodedBf16x8 decoded;
    if (five) {
#pragma unroll
        for (unsigned pair = 0; pair < 4; ++pair) {
            const unsigned low   = (packed >> (pair * 8)) & 0xffU;
            const unsigned shift = pair * 2;
            const int q0 =
                int(((low & 15U) | (((upper >> shift) & 1U) << 4)) ^ 16U) - 16;
            const int q1 =
                int(((low >> 4) | (((upper >> (shift + 1)) & 1U) << 4)) ^ 16U) - 16;
            decoded.pair[pair] =
                __floats2bfloat162_rn(float(q0) * scale, float(q1) * scale);
        }
    } else {
#pragma unroll
        for (unsigned pair = 0; pair < 4; ++pair) {
            const unsigned low = (packed >> (pair * 8)) & 0xffU;
            const int q0       = int((low & 15U) ^ 8U) - 8;
            const int q1       = int((low >> 4) ^ 8U) - 8;
            decoded.pair[pair] =
                __floats2bfloat162_rn(float(q0) * scale, float(q1) * scale);
        }
    }
    auto* output =
        reinterpret_cast<uint4*>(weights + output_group * 32 + subgroup_lane * 4);
    *output = decoded.raw;
}

} // namespace

void q4_q5_gdn_input_cublas_launch(const Tensor& x, const Weight& qk_weight,
                                   const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   cublasHandle_t blas) {
    auto scope = workspace.scope();
    const DeviceSpan weights = workspace.alloc_bytes(std::size_t(kAllRows) * kHidden * 2);
    const DeviceSpan scratch = workspace.alloc_bytes(kGdnBlasWorkspaceBytes);
    // SetStream resets the library's scratch binding; bind arena memory afterwards.
    blas_check(cublasSetStream(blas, stream));
    blas_check(cublasSetWorkspace(blas, scratch.data, scratch.bytes));
    constexpr unsigned kDequantBlocks =
        unsigned((kAllGroups + kDequantGroupsPerBlock - 1) / kDequantGroupsPerBlock);
    dequantize_parents<<<kDequantBlocks, kDequantThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(qk_weight.qdata),
        static_cast<const std::uint16_t*>(qk_weight.scales),
        static_cast<const std::uint8_t*>(value_z_weight.qdata),
        static_cast<const std::uint8_t*>(value_z_weight.qhigh),
        static_cast<const std::uint16_t*>(value_z_weight.scales),
        static_cast<__nv_bfloat162*>(weights.data));
    CUDA_CHECK(cudaGetLastError());
    const float alpha = 1.0f, beta = 0.0f;
    blas_check(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, 10240, x.ne[1], kHidden,
        &alpha, weights.data, CUDA_R_16BF, kHidden, x.data, CUDA_R_16BF, kHidden,
        &beta, qkv.data, CUDA_R_16BF, 10240, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    const auto* z_weights = static_cast<const __nv_bfloat16*>(weights.data) +
                            std::size_t(10240) * kHidden;
    blas_check(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, 6144, x.ne[1], kHidden,
        &alpha, z_weights, CUDA_R_16BF, kHidden, x.data, CUDA_R_16BF, kHidden,
        &beta, z.data, CUDA_R_16BF, 6144, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

} // namespace ninfer::ops::detail
