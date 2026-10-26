# RDNA 4 on a 32-CU part: gfx1200 (RX 9060 XT 16 GB)

This fork's RDNA4 work was developed on **gfx1201** (RX 9070 XT: 64 CU, 32 WGP, 256-bit bus). A
**gfx1200** part (RX 9060 XT: 32 CU, 16 WGP, 128-bit bus) runs the same code objects but is a
different machine: half the wave slots, half the memory bandwidth, half the L2. This document
records what carries over unchanged, what had to be made tunable, the q5_1 V work and its known
issues, and the Windows build requirements for the smaller part.

## What carries over unchanged

Verified on Windows 11, RX 9060 XT (`gfx1200`), through `tools/gfx1200-probe`:

| Assumption in the fork | Measurement on gfx1200 |
| --- | --- |
| `multiProcessorCount` counts WGPs on RDNA4 | **16** for a 32-CU part (16 WGP), so `nsm * 64` = **1024 wave slots** (gfx1201: 2048). The formula scales; no change needed. |
| The launch-size cliff is "the last 32 waves below a multiple of the wave slots" | Same shape on this part: the spike sits exactly on a multiple of 1024, the value `nsm * 64` predicts. |
| Pinned host memory is identity-mapped and coherent (the KV streaming design hands host pointers straight to kernels: the page table, the tag, and the per-layer `ready`/`consumed` flags are host memory that device kernels dereference) | `hipHostMalloc` with `Default`, `Mapped`, and `Mapped\|Coherent` **all** return a device pointer equal to the host pointer; a kernel reads host memory correctly; and a **host write is seen by a kernel polling that address** (the streaming flag path). No pointer translation is needed on this driver. |
| The SDMA copy engine can be picked over HSA | **Not on Windows**: AMD ships no HSA runtime there. `GGML_HIP_HSA_COPY_ENGINE` is forced `OFF` on `WIN32` and the engine uses a dedicated non-blocking stream with events instead. **Verified working** at `-c 131072` with q8_0 K / q4_0 V on Windows/gfx1200. |

### Memory at depth

The 128K allocation itself is not the problem — shallow work under it is not slower than under a
much smaller one. The cost appears as the KV *fills*: with ~11 GiB of weights plus the KV cache,
free VRAM for compute buffers and the MTP draft context runs low at 128K, and that is where the
cliff lives. Streaming the KV to pinned host RAM is what makes a KV larger than the card's free
VRAM possible at all — it buys capacity, not speed. Choose the K/V cache type for depth carefully:
the long-context attention path treats the two block layouts differently, so measure both before
committing to a cache type at depth.

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

The streamed q5_1 path was checked for systematic error against its own non-streaming reference and
against the q4_0 control: a dropped 5th bit or a wrong dim mapping would move perplexity by orders
of magnitude, and it did not.

Two traps found on the way, both **not** caused by the q5_1 work:

* **A 64k no-streaming run with the DFlash2 draft (`-md`) under-measured the draft model's memory**
  (`failed to measure the memory of the extra model, fitting without it`), so the fitted compute
  buffer was borderline and q5_1's slightly larger KV tipped it into a smaller graph. At 16k the
  same config is at parity, and the streamed 64k runs — same kernel — are at parity too. Keep an eye
  on the fit warning when combining `-md` with a large `-c`.
* **`--kv-stream` + a separate draft model (`-md`) deadlocks in decode**: prefill runs to
  completion, then `wait_for: stalled for 30.0 s waiting for the attention to release a slot
  (layer 15)` -> `src/llama-kv-stream.cpp:416: KV streaming stalled`. Reproduced on the *untouched
  q4_0* path, so it is not host RAM exhaustion and it is not the V type. MTP is unaffected because
  the draft and target share one context there. Until that is fixed, pair `--kv-stream` with MTP or
  no speculation, and run the separate-draft configs without streaming.

Still open, in increasing value: **q8_0 V** would reuse the K staging verbatim (34-byte blocks,
signed bytes, no nibble tricks) for 8.5 bits of V at a modest byte cost over q4_0 — better quality
than q5_1, at a higher byte cost.

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
`rocm.nightlies.amd.com/tarball-multi-arch/` index is stale — it still serves clang 23 builds, and
a stale SDK costs a large prefill penalty on gfx1200 (see the bisection note below).

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

### Build determines prefill

The prefill gap seen between this fork and stock on gfx1200 was traced to the **SDK generation, not
the fork's code**: a stale clang 23 SDK costs a large prefill penalty, while a current TheRock 10.3
/ clang 24 build is ahead of stock. Rebuild against a current TheRock tarball before judging any of
these kernels.

### Bisect recipe

```
git worktree add --detach ../llama-base 74bca3950      # the fork's upstream base
scripts/build-windows-gfx1200.sh ../llama-base ../build-base
../build-base/bin/llama-bench.exe -m MODEL -dev ROCm0 -ngl 99 -fa 1 -ctk q5_0 -ctv q5_0 -p 2048 -n 0
```
If the base measures well ahead of the fork on the same machine, the regression is in this fork's
commits and `git bisect` over the 71-commit range with that command as the test finds it. The
engine-driver family to look at first is `vecdotq.cuh` / `mmq-load-tiles.cuh` / `mmq-vec-dot.cuh`
(the quantized dot products the prefill GEMM runs on), then the fused norm/quantize kernels.
