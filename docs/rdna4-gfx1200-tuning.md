# RDNA 4 on a 32-CU part: gfx1200 (RX 9060 XT 16 GB)

This fork's RDNA4 work was developed and measured on **gfx1201** (RX 9070 XT: 64 CU, 32 WGP,
256-bit bus, ~644 GB/s). A **gfx1200** part (RX 9060 XT: 32 CU, 16 WGP, 128-bit bus,
~322 GB/s) runs the same code objects but is a different machine: half the wave slots, half
the memory bandwidth, half the L2. This document records what carries over unchanged, what had
to be made tunable, and the measurements behind both.

## What carries over unchanged

Measured on Windows 11, RX 9060 XT (`gfx1200`, driver 32.0.32015.7029), ROCm 7.15 nightly SDK
(`lib/llvm` clang 23), with `tools/gfx1200-probe`:

| Assumption in the fork | Measurement on gfx1200 |
| --- | --- |
| `multiProcessorCount` counts WGPs on RDNA4 | **16** for a 32-CU part (16 WGP), so `nsm * 64` = **1024 wave slots** (gfx1201: 2048). The formula scales; no change needed. |
| The launch-size cliff is "the last 32 waves below a multiple of the wave slots" | Same shape: a null kernel at 128 threads takes **4.04 us at 1024 blocks** against 2.12 at 1023 and 2.48 at 1025 — the spike sits exactly on a multiple of 1024, which is the value `nsm * 64` predicts. |
| Pinned host memory is identity-mapped and coherent (the KV streaming design hands host pointers straight to kernels: the page table, the tag, and the per-layer `ready`/`consumed` flags are host memory that device kernels dereference) | `hipHostMalloc` with `Default`, `Mapped`, and `Mapped\|Coherent` **all** return a device pointer equal to the host pointer; a kernel reads host memory correctly; and a **host write is seen by a kernel polling that address** (the streaming flag path). No pointer translation is needed on this driver. |
| The SDMA copy engine can be picked over HSA | **Not on Windows**: AMD ships no HSA runtime there. `GGML_HIP_HSA_COPY_ENGINE` is forced `OFF` on `WIN32` and the engine uses a dedicated non-blocking stream with events instead. |

## What had to become tunable

Each of these was a constant measured on a 64-CU part. The default is unchanged, so a gfx1201
run behaves exactly as before; the variable exists to re-measure on a small part.

| Variable | Default | Site | Why it moves on 32 CU |
| --- | --- | --- | --- |
| `GGML_HIP_FA_PAGED_SPLITS_PER_WGP` | `3` | `fattn-paged-decode.cu` (`pd_decode_shape`) | Splits are sized as `3 * nsm` blocks of 8 waves. With 16 WGPs instead of 32 there is half the parallelism, so more splits per WGP may be wanted (or fewer, since each split re-reads partials). |
| `GGML_HIP_MMQ_NARROW_TILES_PER_NSM` | `4` | `mmq.cu` (`ggml_cuda_mul_mat_q`) | The narrow-J override fires when `4 * ntiles < nsm`. Halving `nsm` makes it fire for **twice as many shapes** as on gfx1201. |
| `LLAMA_KV_STREAM_DECODE_MAX` | `96` | `llama-kv-stream.cpp` | The paged-decode vs page-runs crossover (96 query tokens) was measured on a 64-CU part; a 32-CU part saturates the tiled kernel sooner. |

## Windows build requirements

**Use a current ROCm SDK.** The nightlies live at `https://nightly.repo.amd.com/rocm/core/tarball/`
(an HTML page with an embedded `const files = [...]` JSON array); the file for this part is
`therock-dist-windows-gfx120X-all-<version>.tar.gz`, e.g. `10.3.0a20261025`. The older
`rocm.nightlies.amd.com/tarball-multi-arch/` index is stale — it still serves clang 23 builds from
July 2026 and **costs ~1.75x prefill** on gfx1200 (see the measurement below).

With the stale clang 23 SDK, a HIP build also needs a local patch to two headers, because
**VS 18 (MSVC 14.51)** declares the comparison predicates in `<cmath>` as clang builtins and
`<limits>` (included by the ROCm HIP math headers) now pulls `<cmath>` in. Without the patch
*every* HIP translation unit fails with `__device__ function 'isgreater' cannot overload
__host__ __device__ function 'isgreater'`. `scripts/fix-rocm-clang-msvc-cmath.py` applies it
idempotently. **AMD clang 24 (TheRock 10.3) does not need it** — verify with any HIP compile
before spending time on it. The substituted declarations call the same `__builtin_*` and are
usable from device code, so the semantics do not change either way.

Two further requirements, both in the HSA-less path:

* `--rocm-path=<prefix>` (or `HIP_DEVICE_LIB_PATH`) must be given: clang cannot infer the ROCm
  root from a tarball layout where the SDK sits next to Windows' MSVC.
* `hipDeviceProp_t.multiProcessorCount` and the pinned-host-memory results above were verified
  through `tools/gfx1200-probe`, not assumed.

## Measured: qwen35-class hybrid (Qwen3.8-27B IQ3_S, 11.28 GiB, 65 layers, 17 with a KV cache)

`llama-bench`, `-dev ROCm0 -ngl 99 -fa 1 -p 2048`, depth 0, no speculative decoding:

| build | K/V cache | pp2048 t/s | tg128 t/s |
| --- | --- | --- | --- |
| stock (upstream b11860, lemonade nightly, clang 24) | q5_0 / q5_0 | 744.6 | 20.8 |
| this fork, **TheRock 10.3 / clang 24** | q5_0 / q5_0 | **838.5 (+12.6 %)** | **22.4 (+7.6 %)** |
| this fork, TheRock 10.3 / clang 24 | q8_0 / q4_0 | 844.0 | 22.5 |
| this fork, stale TheRock 7.15 / clang 23 | q5_0 / q5_0 | 430.3 (−42 %) | 21.1 |

The fork is ahead of stock on gfx1200 once it is built with a current SDK — with the *user's*
K/V types, not the q8_0/q4_0 the KV-streaming path is specialised for.

Decode is level-to-slightly-ahead; **prefill was ~42 % slower**, and it is *not* this fork's
code. Everything below was measured rather than assumed:

| Hypothesis | Test | Result |
| --- | --- | --- |
| A fork kernel regressed | upstream base `74bca3950` built with the same recipe | **403.6 t/s** — same as the fork, so not the fork |
| The ROCm runtime / kernel packs | this build's code + the stock binary's whole runtime incl. `.kpack` | **396.0 t/s** — not the runtime |
| The KV cache type | sweep q8_0/q4_0, q5_0/q5_0, q5_1/q5_1, f16/f16 | fork 427-430 for **every** type; stock 703-751 — not attention |
| A fork switch | `GGML_CUDA_GDN_MMA=0`, `GGML_CUDA_DISABLE_CONV_FUSION=1`, `GGML_CUDA_DISABLE_ANY_ORDER=1`, `GGML_CUDA_DISABLE_SWIGLU_MMQ=1`, `GGML_HIP_MMQ_NARROW_TILES_PER_NSM=8` | 409-431, i.e. all noise |
| The toolchain | stock binary reports "built with **Clang 24.0.0**"; the SDK from the stale index is **clang 23** | the remaining variable — see the build requirements above |

So: **the SDK generation was the prefill gap**, not the patch set. Rebuild against a current
TheRock tarball before judging any of these kernels on gfx1200.

### Bisect recipe

```
git worktree add --detach ../llama-base 74bca3950      # the fork's upstream base
scripts/build-windows-gfx1200.sh ../llama-base ../build-base
../build-base/bin/llama-bench.exe -m MODEL -dev ROCm0 -ngl 99 -fa 1 -ctk q5_0 -ctv q5_0 -p 2048 -n 0
```
If the base measures ~745 t/s on the same machine, the regression is in this fork's commits and
`git bisect` over the 71-commit range with that command as the test finds it. The engine-driver
family to look at first is `vecdotq.cuh` / `mmq-load-tiles.cuh` / `mmq-vec-dot.cuh` (the
quantized dot products the prefill GEMM runs on), then the fused norm/quantize kernels.
