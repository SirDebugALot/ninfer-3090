# NInfer-3090 v0.6.1 for Windows

This native Windows release supports Qwen3.8-27B, Qwen3.6-27B, and the compact, text-only
Qwen3.6-35B-A3B v0.3.1 artifact. The runtime provides paged KV, concurrent request execution,
compatible-prefix reuse, bounded admission, ReplaySSM, reasoning-effort control, and
OpenAI/Anthropic serving APIs.

## Requirements

- Windows 11 x64;
- GeForce RTX 3090 with a recent NVIDIA driver;
- Microsoft Visual C++ 2022 runtime;
- a supported `.ninfer` model artifact.

The bundled applications and dependency DLLs are native Windows executables. Model artifacts are
not included in the release archive.

## Quick start for Qwen3.8-27B

1. Double-click `download-qwen38.bat`. It downloads the pinned 16.96 GiB container-v2 artifact used by this release, resumes interrupted transfers, and verifies SHA-256.
2. Double-click one launcher:

| Launcher | Profile |
|---|---|
| `run-qwen38-prefill-c6-64k.bat` | Six concurrent requests sharing a 64K RK8V4 cache |
| `run-qwen38-c6-96k-rk8v4.bat` | Six concurrent requests sharing a 96K RK8V4 cache |

Both launchers serve `http://127.0.0.1:8080/v1` with MTP3, LM-head draft, CUDA Graphs, prefix reuse, request logging, and a 2,048-token prefill chunk. The KV capacity is shared across active requests. RK8V4 is experimental and lossy: keys use 8-bit storage and values use 4-bit storage.

The downloader pins Hugging Face revision `18dfc887423fa5aabf3cb56fac41490e462b3fab` and verifies SHA-256 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`. The current unpinned Hugging Face `main` artifact is container v3 with DFlash2 payloads and is not compatible with this container-v1/v2 reader.

## Download the compatible Qwen3.6-35B artifact

The published RTX 3090 measurements use the compact 20.84 GiB container-v1 artifact. Pin its
revision because the Hugging Face repository's unpinned `main` file is now the larger 21.22 GiB
container-v2 artifact with DFlash weights:

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --revision c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c `
  --local-dir models

Get-FileHash .\models\qwen3_6_35b_a3b.ninfer -Algorithm SHA256
```

Expected SHA-256:
`9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163`.

The v0.5 runtime reader accepts both v1 and v2 containers. An error that says only
`artifact magic is not NInfer version 1` comes from an older executable; replace it with the
[v0.5.0 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.5.0-rtx3090).
Although v2 is readable, its DFlash-bearing payload is not the artifact used to qualify the 24 GB
3090 cohort profiles, so pinned v1 remains the recommended download.

## Run the concurrent server

Keep KV capacity explicit on a 24 GB card. Automatic sizing reserves an additional 1 GiB of
headroom and may reject an otherwise viable compact-35B configuration.

```powershell
.\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 4096 --kv-capacity 4096 `
  --max-concurrency 4 --max-pending-requests 32 `
  --prefill-chunk 512 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Prefix reuse is enabled by default; `--no-prefix-reuse` disables it. The server exposes OpenAI
Responses, OpenAI Chat Completions, and Anthropic Messages-compatible endpoints. Run
`.\ninfer-serve.exe --help` for the complete option list.

The compact 35B artifact does not contain DFlash weights. Do not select `--spec dflash`; the
runtime reports the missing optional weights explicitly.

## Qwen3.8-27B C8/8K profile

ReplaySSM reduces speculative GDN state memory enough for the maximum-concurrency Qwen3.8 profile
to use MTP3:

```powershell
.\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --max-concurrency 8 --max-pending-requests 32 `
  --prefill-chunk 1024 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

This 8,192-token shared-pool configuration measured 114.73 aggregate end-to-end tok/s for eight
simultaneous 128-token generations and peaked at 21,818 MiB. Set `--kv-capacity 65536` when all
eight requests need independent 8K capacity; that stronger reservation measured 114.88 tok/s and
23,745 MiB peak. Avoid competing GPU processes.

The cohort size is fixed at startup, but its active membership is not: every decode round compacts
all ready requests into one batch, while completed slots disappear. Follow-up requests wait in the
bounded pending queue and join at a safe round boundary when a lane and memory are available. This
is more predictable than unrestricted dynamic batching because maximum VRAM, workspace, and CUDA
Graph shapes are reserved in advance.

Qwen3.8 supports `low`, `medium`, and `xhigh` reasoning effort. For Chat Completions add the
top-level field `"reasoning_effort": "xhigh"`; Responses uses
`"reasoning": {"effort": "xhigh"}`. The CLI accepts
`--reasoning-effort low|medium|xhigh`.

The paged cache supports BF16, INT8, and experimental opt-in `rk8v4` storage. INT8 remains the
recommended default. On the development RTX 3090, `rk8v4` raised the measured C1 automatic-sizing
boundary from 171,648 to 226,560 tokens with MTP and CUDA Graphs disabled and 1 GiB headroom, but a
matched hard-output test was not quality-equivalent. Use it only after validating your workload.

## Measured 35B capacity

These results used the compact 20.84 GiB artifact on an otherwise idle RTX 3090:

| Workload | Concurrency | Aggregate decode | Peak VRAM |
|---|---:|---:|---:|
| 128 output tokens/request | 1 | 162.7 tok/s | 21.90 GiB |
| 128 output tokens/request | 2 | 267.9 tok/s | 22.21 GiB |
| 128 output tokens/request | 4 | 366.2 tok/s | 22.83 GiB |
| 128 output tokens/request | 6 | 383.4 tok/s | 23.47 GiB |
| 512 output tokens/request | 2 | 399.1 tok/s | within 24 GB |

Concurrency 8 was rejected by admission rather than overcommitting the GPU. Repeating a compatible
26-token prompt reused 24 prefix tokens and reduced measured prefill from 371 ms to 10 ms.

## Build from source

Use Visual Studio 2022, CUDA 12.8 or newer, CMake, and vcpkg:

```powershell
$vcpkgToolchain = 'C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake'

cmake -S . -B build-windows -G 'Visual Studio 17 2022' -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows --config Release --parallel
```

The source rejects unsupported CUDA architectures for this fork. CUDA 13 uses MSVC's conforming
preprocessor automatically.

### Experimental compiler selection for Q4 prefill

Windows/MSVC SM86 single-config Release builds can optionally compile the aligned Q4 SwiGLU
prefill kernel with CUDA 12.6 while retaining the main CUDA toolkit, runtime, and device linker.
From an initialized Visual Studio x64 developer environment, add this CMake cache option to a
Ninja Release configuration alongside its existing toolkit/vcpkg settings:

```text
-DNINFER_Q4_FULL_NVCC="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
```

The option is empty/off by default and is not a system CUDA downgrade. Only Q4 calls with at least
1,024 tokens and a token count divisible by 128 use the separately named CUDA 12.6 kernel. Short
calls, decode/MTP, and all tail shapes retain the main compiler because the CUDA 12.6 tail kernel
was slower. The aligned auxiliary kernel now uses direct MMA weight fragments and a WN32 tile;
it preserves the represented arithmetic, storage, and zero-workspace contract, but is no longer
just a separately compiled copy of the main kernel. See the
[short RTX 3090 prefill summary](rtx-3090-faster-prefill.md) for the retained implementation and evidence.
The auxiliary object uses the inherited MSVC environment and a header depfile for incremental
rebuilds; changing its headers regenerates it. Clear `NINFER_Q4_FULL_NVCC` to disable the route.
Qualify the public Q4 Op and end-to-end prefill on the selected hardware before deploying it.

## Release validation

The v0.5 Windows release gate rebuilt `ninfer.exe`, `ninfer-serve.exe`, and `ninfer_bench.exe`,
loaded the official Qwen3.8 artifact, generated coherent output, and completed C1-C4 plus C8/8K
serving checks. Focused tests cover artifact reading/materialization, request memory, admission,
paged KV, prefix append, speculative rounds, and the relevant SM86 W8 linear paths.
