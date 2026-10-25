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
| The SDMA copy engine can be picked over HSA | **Not on Windows**: AMD ships no HSA runtime there. `GGML_HIP_HSA_COPY_ENGINE` is forced `OFF` on `WIN32` and the engine uses a dedicated non-blocking stream with events instead. **Verified working**: `--kv-stream --kv-resident-mib 1024 --kv-resident-mib-decode 1536` at `-c 131072` with q8_0 K / q4_0 V loads and serves on Windows/gfx1200 (prefill 381.6 t/s against 419.3 not streaming at the same 2k depth, no stall, no timeout). |

### Context depth is not free on a 16 GB card

Shallow work under a 128K allocation is *not* slower than under a 36K one, so the big allocation
itself is fine (server harness, 2,025-token prompt, q5_0/q5_0, MTP on):

| `-c` | prefill t/s | decode t/s |
| --- | --- | --- |
| 36864 | 418.0 | 47.7 |
| 69632 | 413.6 | 48.1 |
| 135168 | 419.3 | 47.3 |

The cost appears as the KV *fills*: the same config falls from 30.1 t/s decode at 92k to 8.2 t/s
at 131k (prefill 469 -> 176). With 11.28 GiB of weights plus 2.92 GiB of KV at q5_0/q5_0 there is
about 1.2 GiB left for compute buffers and the MTP draft context, which is where the cliff lives.

### The fix for that cliff is this fork's own `--kv-stream`

Bounding the resident KV and streaming the rest from host RAM buys back the whole prefill loss at
128K. Server harness, 131,072-token prompt, MTP + ngram-mod, ROCm0, same flags except the pool:

| arm | prefill t/s | decode t/s |
| --- | --- | --- |
| q8_0 K / q4_0 V, no streaming (3.45 GiB resident) | 250.2 | 20.02 |
| **q8_0 K / q4_0 V, `--kv-stream --kv-resident-mib 1024 --kv-resident-mib-decode 1536`** | **531.8 (+113 %)** | **21.40** |
| q5_0 / q5_0, no streaming | 176.0 | 8.24 |

Two things to note:

* `--kv-stream` requires **q8_0 K and q4_0 V** (`llama-kv-cache.cpp` refuses anything else, and the
  paged kernels in `fattn-paged-decode-kernel.cuh` / `fattn-prefill-rdna4.cuh` hardcode those block
  layouts). Supporting a Q5_0 V cache there is the work item that would combine better V quality
  with streaming: the K side needs nothing, only the V dequant and the type guards.
* q5_0/q5_0 is *worse than* q8_0/q4_0 at 128K without streaming (8.2 against 20.0 t/s) despite
  being a **smaller** cache, so the long-context FA path evidently treats the two type combinations
  differently. Worth measuring before choosing a cache type for depth.

### The tunables: measured at 128K, defaults win

All at `-c 131072 --kv-stream --kv-resident-mib 1024 --kv-resident-mib-decode 1536`, q8_0/q4_0,
MTP + ngram-mod (decode samples are 26 tokens — the prompt plus 128 generation tokens overflows the
slot, so the server truncates; prefill numbers are over the full 131,033 tokens):

| arm | prefill t/s | decode t/s |
| --- | --- | --- |
| defaults (`splits` 3, `decode_max` 96) | 530.5 | 20.25 |
| `GGML_HIP_FA_PAGED_SPLITS_PER_WGP=6` | 529.2 | **16.73** — more splits re-read partials, keep 3 |
| `LLAMA_KV_STREAM_DECODE_MAX=160` | faults at init | needs more ring residency than the card has |
| `LLAMA_KV_STREAM_DECODE_MAX=64` + `splits=6` | faults at init | as above |

Both failing arms die with `ROCm error: unspecified launch failure` inside
`ggml_backend_cuda_buffer_set_tensor` / `hipStreamSynchronize` (ggml-cuda.cu:861) within half a
second of the first compute, i.e. an over-committed allocation rather than a kernel bug. Resident
pool `2048/3072` fails the same way. **The 64-CU defaults transfer to this part unchanged** — the
knobs are there to re-measure, not because they needed moving.

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
