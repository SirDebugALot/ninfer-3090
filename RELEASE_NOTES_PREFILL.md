# NInfer RTX 3090 - Faster Prefill

This release is based on `v0.6.1-rtx3090` and contains the accepted Round 11
prefill implementation for Qwen3.8-27B on RTX 3090/SM86.

## Highlights

- A fresh-process A/B/B/A comparison at 32K input measured 785.03 tok/s for
  the official v0.6.1 Windows binary and 1037.43 tok/s for Round 11 (+32.15%).
- The strict comparison uses INT8 KV because the official v0.6.1 benchmark
  does not support RK8V4. RK8V4 remains available in the optimized release.
- Retained changes focus on Q4 SwiGLU decoded-weight reuse, Q4/Q5 projection
  scheduling, cuBLAS-backed large shapes and SM86 attention/dequantization.
- Generated malformed UTF-8 is logged and repaired per request so it does not
  take the serving worker down. This is a stability fix, not a speed claim.
- The Windows package includes C6 64K and C6 96K RK8V4 launchers on port 8080.
- The model downloader pins and verifies the compatible Qwen3.8 container-v2 artifact instead of following the mutable Hugging Face `main` branch.

Results are specific to the tested RTX 3090, Qwen3.8-27B groupwise-int model,
32K input, INT8 KV, 64K shared capacity, chunk 2048, MTP3 and CUDA Graph setup.
Model weights are not included.
