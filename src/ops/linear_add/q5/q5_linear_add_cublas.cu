#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "core/device.h"
#include "core/weight.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Q5 LinearAdd cuBLAS status " + std::to_string(int(status)));
    }
}

__global__ void dequantize_q5(const std::uint8_t* codes, const std::uint8_t* high,
                            const std::uint16_t* scales, __nv_bfloat162* weights,
                            std::size_t pairs) {
    for (std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pairs; pair += std::size_t(gridDim.x) * blockDim.x) {
        const std::size_t group = pair >> 5;
        const unsigned lane = unsigned(pair) & 31U;
        const unsigned low = codes[pair];
        const unsigned upper = high[group * 8 + (lane >> 2)];
        const unsigned shift = (lane & 3U) * 2U;
        const int q0 = int(((low & 15U) | (((upper >> shift) & 1U) << 4)) ^ 16U) - 16;
        const int q1 = int(((low >> 4) | (((upper >> (shift + 1)) & 1U) << 4)) ^ 16U) - 16;
        const float scale = __half2float(__ushort_as_half(scales[group]));
        weights[pair] = __floats2bfloat162_rn(float(q0) * scale, float(q1) * scale);
    }
}
} // namespace

void q5_linear_add_cublas_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                 WorkspaceArena& ws, cudaStream_t stream, cublasHandle_t blas) {
    auto scope = ws.scope();
    const std::size_t pairs = std::size_t(w.n) * w.k / 2;
    auto weights = ws.alloc_bytes(pairs * sizeof(__nv_bfloat162));
    auto scratch = ws.alloc_bytes(kQ5BlasWorkspaceBytes);
    // SetStream resets cuBLAS workspace; always bind explicit arena memory afterwards.
    blas_check(cublasSetStream(blas, stream));
    blas_check(cublasSetWorkspace(blas, scratch.data, scratch.bytes));
    const unsigned blocks = unsigned(std::min<std::size_t>((pairs + 255) / 256, 65535));
    dequantize_q5<<<blocks, 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint16_t*>(w.scales),
        static_cast<__nv_bfloat162*>(weights.data), pairs);
    CUDA_CHECK(cudaGetLastError());
    const float alpha = 1.0f, beta = 1.0f;
    blas_check(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, w.n, x.ne[1], w.k,
        &alpha, weights.data, CUDA_R_16BF, w.k, x.data, CUDA_R_16BF, w.k,
        &beta, residual.data, CUDA_R_16BF, w.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
} // namespace ninfer::ops::detail
