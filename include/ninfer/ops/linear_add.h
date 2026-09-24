#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are Q5G64_F16S RowSplit
 *   [5120,17408] or [5120,6144], W8G32_F16S RowSplit [2048,4096] or [2048,6144], and NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], or BF16_CTRL Contiguous [5120,6144].
 *   T may be any positive value.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16 or A4 criterion, not the oracle. Production routes may fuse or
 *   materialize the projection and may choose their natural accumulator, activation quantization,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *
 * Compute policy:
 *   Q5, W8, and BF16_CTRL admit only A16Only. NVFP4 admits A16Only and AllowA4; AllowA4 permits
 *   the private resolver to select a qualified A16 or A4 route at every positive T.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. There is no model-persistent state side effect. The capacity query covers execution
 *   with or without the optional cuBLAS resource.
 *
 * Execution:
 *   `blas` may be null. A non-null handle is caller-owned (normally DeviceContext::blas), created
 *   before capture on the current CUDA device, and used only with streams on that device. The
 *   caller configures CUBLAS_POINTER_MODE_HOST, CUBLAS_ATOMICS_NOT_ALLOWED, and
 *   CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION; this Op uses BF16 operands with FP32
 *   accumulation and reduction, not a TF32/FAST compute mode. The Op updates the handle's stream
 *   and workspace binding. Submissions through the handle must be serialized on the declared
 *   stream, never concurrent from another thread or stream. No handle is created lazily here.
 *   Keep the owning context and caller-owned workspace alive until queued work and any captured
 *   graph replays complete; captured operand and workspace addresses remain stable.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream, cublasHandle_t blas = nullptr);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream, cublasHandle_t blas = nullptr);

} // namespace ninfer::ops
