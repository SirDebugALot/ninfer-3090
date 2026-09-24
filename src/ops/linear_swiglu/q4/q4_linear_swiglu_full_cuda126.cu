#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/token_slices.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// This direct-fragment route intentionally does not use GemmCfg's generic
// shared-memory estimate: it has no decoded-weight As tile. Four two-warp M
// groups share one activation tile, so a CTA produces 64 gate/up pairs while
// retaining two-block residency on SM86.
struct FullCfg {
    static constexpr int BM         = 128;
    static constexpr int BN         = 128;
    static constexpr int BK         = 64;
    static constexpr int THREADS    = 256;
    static constexpr bool CG_LOAD   = true;
};

__device__ __forceinline__ unsigned decode_fragment_pair(unsigned packed, float scale) {
    const float q0 = float(int((packed & 15U) ^ 8U) - 8);
    const float q1 = float(int((packed >> 4) ^ 8U) - 8);
    const __nv_bfloat162_raw pair = __floats2bfloat162_rn(q0 * scale, q1 * scale);
    return unsigned(pair.x) | (unsigned(pair.y) << 16);
}

// The admitted profile has K=padded_K=5120 and 17408 gate/up pairs. Packed
// weights are staged once per CTA, then decoded directly into MMA A registers.
// Four M warp groups each fold 16 gate/up pairs. A warp reuses decoded weights
// across 64 tokens, with the same K reduction order. FP16 scales, FP32 products,
// BF16 operands, and the fused FP32 epilogue remain.
__global__ __launch_bounds__(FullCfg::THREADS, 2)
void q4_linear_swiglu_direct_fragment_cuda126_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate, int k, int tokens) {
    using Cfg = FullCfg;
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK;
    constexpr int WN = 64, S = 2, MT = 2, NT = WN / 8, WarpsN = BN / WN;
    constexpr int Threads = Cfg::THREADS;
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * 32];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * 4];
    const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int warp_m = warp / WarpsN, warp_n = warp % WarpsN;
    const int gid = lane >> 2, lid = lane & 3;
    // Visit a small row group across token tiles before advancing to the next
    // rows. The admitted shape has 272 row tiles, divisible by this group size.
    // This changes only CTA order, shortening packed-weight reuse distance in L2.
    constexpr int GroupM = 4;
    const int linear = int(blockIdx.x) + int(blockIdx.y) * int(gridDim.x);
    const int group_span = GroupM * int(gridDim.y);
    const int m_tile = (linear / group_span) * GroupM + linear % GroupM;
    const int t_tile = (linear % group_span) / GroupM;
    const int m0 = m_tile * (BM / 2), t0 = t_tile * BN;
    const int groups = k / 64, nkt = k / BK;
    float acc[MT][NT][4]{};

    auto folded_global_row = [&](int row) {
        const int pair = row >> 5;
        const int inner = row & 31;
        return m0 + pair * 16 + (inner & 15) + (inner >> 4) * intermediate;
    };
    auto stage_load = [&](int stage, int kt) {
#pragma unroll 1
        for (int c = tid; c < BN * (BK / 8); c += Threads) {
            const int token = c / (BK / 8), kl = (c % (BK / 8)) * 8;
            gemm_cp_async<16, Cfg>(&Bs[stage][token * BK + gemm_swz64(token, kl)],
                                  &x[std::int64_t(t0 + token) * k + kt * BK + kl]);
        }
#pragma unroll 1
        for (int c = tid; c < BM * 2; c += Threads) {
            const int row = c >> 1, half = c & 1;
            const std::int64_t group = std::int64_t(folded_global_row(row)) * groups + kt;
            gemm_cp_async<16, Cfg>(&Cr[stage][row * 32 + half * 16],
                                  &codes[group * 32 + half * 16]);
        }
#pragma unroll 1
        for (int row = tid; row < BM; row += Threads) {
            const std::int64_t aligned_group =
                std::int64_t(folded_global_row(row)) * groups + (kt & ~1);
            gemm_cp_async<4, Cfg>(&Sr[stage][row * 4], &scales[aligned_group * 2]);
        }
    };
#pragma unroll
    for (int s = 0; s < S; ++s) {
        stage_load(s, s);
        cp_commit();
    }
    for (int it = 0; it < nkt; ++it) {
        const int stage        = it % S;
        const int scale_offset = (it & 1) * 2;
        cp_wait<S - 1>();
        __syncthreads();
        float scale0[MT], scale1[MT];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = warp_m * 32 + mi * 16 + gid;
            scale0[mi] = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(&Sr[stage][row * 4 + scale_offset])));
            scale1[mi] = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(
                    &Sr[stage][(row + 8) * 4 + scale_offset])));
        }
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            unsigned af[MT][4], bf[NT][2];
            const int pair_k = ki * 8 + lid;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_m * 32 + mi * 16 + gid;
                // m16n8k16 A covers rows gid/gid+8 and column pairs
                // 2*lid / 2*lid+8 within each sixteen-wide K slice.
                af[mi][0] = decode_fragment_pair(Cr[stage][row * 32 + pair_k], scale0[mi]);
                af[mi][1] =
                    decode_fragment_pair(Cr[stage][(row + 8) * 32 + pair_k], scale1[mi]);
                af[mi][2] =
                    decode_fragment_pair(Cr[stage][row * 32 + pair_k + 4], scale0[mi]);
                af[mi][3] =
                    decode_fragment_pair(Cr[stage][(row + 8) * 32 + pair_k + 4], scale1[mi]);
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_n * WN + ni * 8 + (lane & 7);
                const int col = ki * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][row * BK + gemm_swz64(row, col)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                }
            }
        }
        __syncthreads();
        if (it + S < nkt) { stage_load(stage, it + S); }
        cp_commit();
    }
#pragma unroll
    for (int mi = 0; mi < MT / 2; ++mi) {
        const int r0 = m0 + warp_m * 16 + mi * 16 + gid, r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c0 = t0 + warp_n * WN + ni * 8 + 2 * lid, c1 = c0 + 1;
            out[std::int64_t(c0) * intermediate + r0] =
                __float2bfloat16_rn(silu(acc[mi][ni][0]) * acc[mi + MT / 2][ni][0]);
            out[std::int64_t(c1) * intermediate + r0] =
                __float2bfloat16_rn(silu(acc[mi][ni][1]) * acc[mi + MT / 2][ni][1]);
            out[std::int64_t(c0) * intermediate + r1] =
                __float2bfloat16_rn(silu(acc[mi][ni][2]) * acc[mi + MT / 2][ni][2]);
            out[std::int64_t(c1) * intermediate + r1] =
                __float2bfloat16_rn(silu(acc[mi][ni][3]) * acc[mi + MT / 2][ni][3]);
        }
    }
}

} // namespace

void q4_linear_swiglu_cuda126_full_launch(const Tensor& x, const Weight& weight,
                                          Tensor& out, cudaStream_t stream) {
    using Cfg = FullCfg;
    // The caller admits only T >= 1024 divisible by 128. Slice capacity is also
    // a multiple of 128, so every slice preserves the full-tile precondition.
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice = out.slice(1, offset, count);
        const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], Cfg::BM / 2)),
                        static_cast<unsigned>(count / Cfg::BN));
        q4_linear_swiglu_direct_fragment_cuda126_kernel
            <<<grid, Cfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x_slice.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out_slice.data), out.ne[0], x.ne[0], count);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
