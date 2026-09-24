# NInfer RTX 3090: faster prefill

This branch is based on NInfer-3090 `v0.6.1-rtx3090` and targets NVIDIA
GA102/SM86. The performance baseline is the original v0.6.1 Windows binary;
the selected implementation is the accepted Round 11 build. Later experimental
performance candidates are not included. Generated-UTF-8 recovery is included
as a serving-stability fix, not as a speed claim.

## Result

Qwen3.8-27B groupwise-int, one active request, 32,768 input tokens, RTX 3090,
INT8 KV, 65,536-token shared context/KV capacity, 2,048-token prefill chunks,
MTP3, optimized LM-head draft and CUDA Graphs. Each observation used a fresh
process in A/B/B/A order:

| Build | Prefill |
|---|---:|
| Official v0.6.1 Windows binary | 791.83, 778.22 tok/s; **785.03 mean** |
| Accepted Round 11 build | 1039.13, 1035.73 tok/s; **1037.43 mean** |

The direct improvement is **+32.15%**. Both builds used the same model, corpus,
workspace (361,906,176 bytes) and benchmark arguments. No thermal-slowdown event
was reported. The official v0.6.1 benchmark accepts only BF16 or INT8 KV, so a
strict original-versus-Round-11 RK8V4 comparison is impossible. RK8V4 remains
the recommended long-context serving profile, but it is not used for this
headline A/B. Results are specific to this GPU, model and workload.

The compared SHA-256 values were `e1346064890428485291524ce19e4a47e85ebcb6fe0b5c281ed2d5e026c83b0c`
for the official benchmark and `eb4683bf7526fb9f38e097799177cfb709a8086cb1dd767a236417635105797c`
for Round 11.

## What changed

- The aligned Q4 SwiGLU prefill kernel uses a CUDA 12.6 SM86 specialization.
  Its warp tile reuses each decoded weight fragment across more token columns,
  reducing packed-weight decode work while preserving the represented math.
- Large Q4/Q5 projection shapes use qualified target-specific schedules,
  including dequantization plus cuBLAS where it was faster than the original
  custom route.
- Cached attention and Q4/Q5 dequantization were retuned for SM86 register,
  cache and warp-level reuse behavior.
- Numerical operator tests, CUDA Graph replay checks and a packaged server
  smoke test passed. Workspace for the final 32K comparison was unchanged.

The model format, stored weights, KV codec, MTP3 and CUDA Graph policy were not
changed to obtain the speedup. Decode performance is not the headline claim.

## Build requirements

- Windows 11 x64 and Visual Studio 2022 C++ tools
- CMake and Ninja
- CUDA 13.2 for the main build
- CUDA 12.6 for the selected aligned Q4 translation unit
- vcpkg dependencies from the committed `vcpkg.json`

The GitHub Actions workflow installs both CUDA toolkits, builds `sm_86`, creates
a Windows ZIP and publishes it as a workflow artifact. A pushed release tag also
attaches the ZIP and checksum to GitHub Releases. Compilation does not require a
GPU, but the resulting binary still needs an RTX 3090 and a compatible NVIDIA
driver for runtime validation.

## Runtime profile

The release includes `run-qwen38-prefill-c6-64k.bat` and
`run-qwen38-c6-96k-rk8v4.bat`. Both start the OpenAI-compatible server on port
8080 with concurrency 6, RK8V4, chunk 2048, MTP3, LM-head draft and CUDA
Graphs. The launchers provide shared KV capacities of 64K and 96K respectively.
Use `download-qwen38.bat` to obtain and verify the pinned compatible container-v2
Qwen3.8 artifact; GGUF and the current container-v3 Hugging Face artifact are
not compatible with this release.
