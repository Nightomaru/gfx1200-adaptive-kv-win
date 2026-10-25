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

### 128K: what the cache type and the streaming pool are actually worth

Firm numbers — full **128-token** decode sample (`-c 132096`, so the 131,033-token prompt plus the
generation cannot overflow the slot), MTP + ngram-mod, ROCm0:

| arm at 128K | prefill t/s | decode t/s |
| --- | --- | --- |
| q8_0 K / q4_0 V, no streaming | 489.3 | **36.82** |
| q8_0 K / q4_0 V, `--kv-stream` 1024/1536 | **528.0** | 34.53 |
| q5_0 / q5_0, no streaming (92k) | 431.0 | 23.27 |
| q5_0 / q5_0, no streaming (128k) | 176.0 | 8.24 |

Two corrections to earlier, shorter measurements:

* **Streaming is worth ~+8 % prefill and ~-6 % decode at 128K, not +113 %.** The +113 % came from a
  *26-token* decode sample: the first tokens after a 131k prefill carry the warm-up, and the
  no-stream arm is itself unstable run to run (prefill measured 250 then 489 t/s on identical
  flags). Never conclude from a truncated decode sample, and repeat the no-stream config.
* **The cache type dominates at depth.** q8_0/q4_0 holds ~35-37 t/s decode at 128K where q5_0/q5_0
  craters to 8.2 — despite q5_0/q5_0 being the *smaller* cache (2.92 GiB against 3.45). Whatever
  the long-context FA path does with the two block layouts differs a lot; measure both before
  choosing a cache type for depth.

Repeats (two runs each, so the numbers above are not a one-off):

| arm, 128K, 128-token samples | prefill t/s | decode t/s |
| --- | --- | --- |
| no streaming | 486.2, 504.0 → **495.1** | 39.16, 38.66 → **38.91** |
| `--kv-stream` 1024/1536 | 530.7, 530.3 → **530.5** | 35.65, 36.30 → **35.98** |

So streaming is a **+7 % prefill / -7 % decode** trade at 128K on this part — it is a capacity
feature (it is what makes a KV larger than the card's free VRAM possible at all, e.g. 256K or a
bigger model), not a speed fix for 128K with q8_0/q4_0, because that combination already fits.

Both arms beat the stock 92k reference (431.0 / 23.27) while running 39 % more context: +15 to +23 %
prefill and +55 to +67 % decode.

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

## Work item: a better V cache under `--kv-stream`

`--kv-stream` takes q8_0 K and q4_0 V only. The K side is already q8_0 (what the user wanted), so
the limit is the V side. Two routes, in increasing cost:

**Route 1 — q8_0 V (reuse the K path verbatim).** `PD_Q8_BLK = 34` is what the *K* staging already
handles, including the scale read at `krow + blk*PD_Q8_BLK`. A q8_0 V variant reuses that dequant
unchanged and only needs the V layout constants and the two scale-gather sites adjusted
(`PD_V_ROW` 144 -> 272, `PD_V_VEC` 9 -> 17, `PD_V_LOADS`, the `warp * PD_Q4_BLK` offsets) plus a
signed-byte fragment gather in place of the nibble tricks. Cache cost at 128K: 4.13 GiB against
3.45 for q4_0 — and **8.5 bits of V against 4.5**, i.e. better quality than the q5_0 the user asked
for, for about 0.7 GiB.

**Route 2 — q5_0 V (faithful to the request, more work).** Q5_0's block is 22 bytes (2 d + 4 qh +
16 qs) and the row is 176 B = 11 x 16 B pieces, both still 16 B aligned, so the staging shape
survives — but the 5-bit unpack needs its own path: the existing gather is written around Q4_0's
nibble layout ("high nibble, dt 1") and the piece->block scale trick at `PD_V_ROW` byte 2*v no
longer lines up. Cache cost at 128K: 3.72 GiB with q8_0 K.

Either route also needs the two guards relaxed: `src/llama-kv-cache.cpp:178` (the throw) and
`ggml/src/ggml-cuda/fattn-paged-decode.cu:39` (the `supports` test), plus the same treatment in
`fattn-prefill-rdna4.cuh` for the paged prefill path.

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
