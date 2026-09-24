#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
// Private full-tile entry, linked only when NINFER_Q4_FULL_NVCC is configured.
void q4_linear_swiglu_cuda126_full_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
