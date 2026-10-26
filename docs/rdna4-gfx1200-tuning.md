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

### Host memory: what streaming costs

Streaming keeps the whole KV in **pinned (non-pageable) host RAM** — there is no way around it, the
host store is where evicted pages live. At 128K with q8_0/q4_0 that is ~3.45 GiB, plus the small
device-slot ring. On this 16 GB machine free host RAM measured **9.4-9.5 GiB** around every arm of
the repeat runs (sampled before and after each), so the pool fits with room to spare — and that also
rules out host-RAM exhaustion as the cause of the one 250 t/s prefill outlier seen earlier. Without
streaming the KV sits in VRAM and costs host RAM nothing, which is the other half of the trade:
streaming buys capacity, not speed, at 128K.

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

## Done: a q5_1 V cache under `--kv-stream`

`--kv-stream` now takes **q8_0 K with q4_0 (V_TAG 0) or q5_1 (V_TAG 1) V**, selected per kernel
instantiation so the q4_0 path compiles to exactly what it did before. The Q5_1 block is 24 bytes
(`d, m, qh[4], qs[16]`), so a V row is 192 B = 12 x 16 B pieces — still 16 B aligned, but the Q4_0
staging trick of reading block *v*'s scale out of 16-byte piece *v* breaks (the pair lives at byte
24*b), which is where all the work was:

* `fattn-paged-decode-kernel.cuh` — V_TAG picks the geometry; `PD_Q51_PIECE` maps a 16 B piece to
  the block whose `d`/`m` pair it carries (byte 0 = x, byte 8 = z, low half = d, high half = m);
  the 5th bit is gathered from the block's `qh` mask and `V = d*(nibble|bit5<<4) + m` via a new
  `pd_u5x2` (half(1024 + q5) - 1024, no Q4_0-style -8 bias).
* `fattn-prefill-rdna4.cuh` — `ppq_load`/`ppq_store` on `V51`: the same `half(1024 + nibble)`
  idiom carries the 5th bit, so the offset moves 1032 -> 1024 and `m` is added after the scale;
  `d`/`m`/`qh` read as one 8-byte load per block.
* `fattn-paged-decode.cu` — the type gate takes q5_1 and both launches branch on `V->type`.
* `src/llama-kv-cache.cpp:178` — the `--kv-stream` throw now takes a q4_0 *or* q5_1 V cache.

Correctness (llama-perplexity, 8 x 2048-token chunks of the fork's own docs):

| arm | PPL |
| --- | --- |
| q5_1 V, stock FA path (no streaming) — reference | 3.9998 |
| q5_1 V, `--kv-stream` (the new paged kernels) | 3.9946 |
| q4_0 V, `--kv-stream` (existing path, control) | 4.0002 |

The streamed q5_1 result sits 0.13 % from its own reference — the same magnitude as the
q4_0-vs-q5_1 *quantisation* difference (0.14 %) — so there is no systematic error; a dropped 5th
bit or a wrong dim mapping would move PPL by orders of magnitude more.

Speed, 65,536-token prompt + 128 generated, `draft-dflash,ngram-mod --spec-draft-n-max 6`:

| 16k context (no VRAM pressure) | prefill t/s | decode t/s |
| --- | --- | --- |
| q8_0 / q4_0, no streaming | 654.9 / 650.0 | 79.3 / 78.9 |
| q8_0 / q5_1, no streaming | 654.1 / 650.4 | 75.9 / 76.3 |
| q8_0 / q4_0, `--kv-stream` | 710.9 / 705.3 | 21.1 / 21.1 |
| q8_0 / q5_1, `--kv-stream` | 706.8 / 707.2 | 21.0 / 20.9 |

| 64k context, `--kv-stream` | prefill t/s | decode t/s |
| --- | --- | --- |
| q8_0 / q4_0 | 657.4 / 652.6 | 18.7 / 18.3 |
| q8_0 / q5_1 | 654.0 / 653.0 | 18.1 / 18.3 |

So: prefill parity everywhere, decode −1 % under streaming and −3-4 % without it — and the KV cost
of q5_1 is ~+11 % of bytes (24 against 18 per block) for 5.5 bits of V against 4.5.

Two traps measured on the way, both **not** caused by the q5_1 work:

* **A 64k no-streaming run with the DFlash2 draft showed q5_1 prefill at 279 t/s against q4_0's
  610.** That is a *fit* artefact, not the kernels: with `-md` the fitter cannot measure the draft
  model's memory (`failed to measure the memory of the extra model, fitting without it`), so the
  fitted compute buffer is borderline and q5_1's slightly larger KV tips it into a smaller graph.
  At 16k the same config is at parity (table above), and the streamed 64k runs — same kernel —
  are at parity too. Keep an eye on the fit warning when combining `-md` with a large `-c`.
* **`--kv-stream` + a separate draft model (`-md`) deadlocks in decode**: prefill runs to
  completion, then `wait_for: stalled for 30.0 s waiting for the attention to release a slot
  (layer 15)` -> `src/llama-kv-stream.cpp:416: KV streaming stalled`. Reproduced on the *untouched
  q4_0* path at 64k and 128k with 6.5 GiB of free host RAM and under 1 GiB pinned — it is not host
  RAM exhaustion, and it is not the V type. MTP is unaffected because the draft and target share
  one context there. Until that is fixed, pair `--kv-stream` with MTP or no speculation, and run
  the separate-draft configs without streaming.

Still open, in increasing value: **q8_0 V** would reuse the K staging verbatim (34-byte blocks,
signed bytes, no nibble tricks) for 8.5 bits of V at ~+0.7 GiB over q4_0 at 128k — better quality
than q5_1, at a higher byte cost.

### Draft depth beats draft length: `--spec-draft-n-max 4`, not 6

DFlash2 + `ngram-mod` at 64k, q5_1 V streamed, 2 runs per arm (128 generated tokens; prefill is
~620 t/s in every arm, so only decode moves):

| draft settings | decode t/s | acceptance | mean accepted |
| --- | --- | --- | --- |
| `--spec-draft-n-max 4` + upstream `ngram-mod` 32/12/24 | **39.80** (38.1 / 41.5) | 0.831 | 6.68 |
| `--spec-draft-n-max 4`, default `ngram-mod` | 38.96 (39.2 / 38.7) | 0.831 | 7.00 |
| `--spec-draft-n-max 6`, default `ngram-mod` | 36.00 (36.1 / 35.9) | 0.782 | 8.40 |
| `--spec-draft-n-max 6` + upstream `ngram-mod` | 33.34 (33.4 / 33.3) | 0.736 | 7.41 |

The mechanism is in the acceptance counters: at n-max 6 the *rate* drops (0.78/0.74 against 0.83)
while the mean accepted length gains little (7.4-8.4 against 6.7-7.0) — the deeper draft positions
are mostly rejected, so every verify pass carries more query rows for near-nothing. n-max 4 is
worth **+9-11 % decode** over 6.

The upstream `ngram-mod` overrides (n-max 32 / n-min 12 / n-match 24) are a wash at n-max 4 and
actively hurt at n-max 6 on this workload — but the workload is *varied* prose and code, not the
repetitive content long ngram matches are for. Re-measure on a real code-editing session before
dropping them.

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
