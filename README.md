# llama.cpp (gfx1200-adaptive-kv-win)

**Adaptive KV streaming + rdna-boosts, merged — on native Windows for AMD RDNA 4, built and tuned for the Radeon RX 9060 XT (gfx1200, 16 GB).**

A standalone llama.cpp tree combining three lines of work into one Windows/gfx1200 build:

- [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm), branch `feature/adaptive-kv-stream-rocm` — adaptive KV streaming plus RDNA 4 ROCm optimizations (developed on the RX 9070 XT, `gfx1201`, Linux)
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) v16 — the RDNA performance patch series (WMMA flash attention, chunked gated-delta-net prefill, k-quant decode boosts)
- this repo's native Windows port, merge work, and fixes

The result: a 16 GB card runs ~27B-class hybrid (gated-delta-net) models at long context with the KV cache paged over PCIe at full attention quality, with DFlash2 speculative decoding and a streamed **q8_0 K / q5_1 V** cache.

---

## What this build changes

- **rdna-boosts v16 merged into the KV-stream tree.** The two lines were replayed and merged with explicit merge-repair commits; where the forks disagreed on kernel/op semantics, v16's conventions win and the fork code was reconciled to them (v16 MMA gated-delta-net kernel restored, fork-only ReplaySSM GPU plumbing dropped, mmq/vecdotq micro-optimizations reverted to the v16 forms).
- **Fixed a merge bug that corrupted all qwen35-family output.** The CPU gated-delta-net kernel read the ReplaySSM record flag from op parameter 1 — which v16's plain gated-delta-net op uses for a batch count — so every plain call ran the wrong path. The flag is now derived from the op's tensor inputs (the rec tensor, `src[6]`), which only the replay op sets. Quant-independent, and cured on CPU and GPU alike; verified over the qwen35/GDN discriminator set plus a non-GDN control.
- **Native Windows port of the KV-stream stack.** The upstream copy engine assumes the Linux HSA runtime, which AMD does not ship on Windows. Ported to a HIP stream+event copy path behind `GGML_HIP_HSA_COPY_ENGINE`, after verifying that `hipHostMalloc` is identity-mapped and host-write-coherent on gfx1200/WDDM — so the host-pointer page design works without modification.
- **gfx1200 as a first-class target.** Built for `GPU_TARGETS=gfx1200` (32 CU / 16 WGP); the RDNA4 launch-size heuristics written for the 64-CU 9070 XT were re-measured on the smaller part.
- **q8_0 K / q5_1 V KV streaming.** Paged decode *and* prefill kernel variants for Q5_1's 24-byte block (5-bit unpack plus per-block offset), selected at compile time so the q4_0 path compiles unchanged.
- **DFlash2 speculative draft.** `--spec-type draft-dflash,ngram-mod` with a separate draft GGUF (`--model-draft`). Note: ReplaySSM draft replay (`--spec-rs-replay`) is **not** GPU-supported in this tree — the replay op has no HIP kernel here — so it stays out of the serve config.
- **A current TheRock ROCm.** Built with a Windows `gfx120X` [TheRock](https://github.com/ROCm/TheRock) tarball (AMD clang 24). Mind the SDK staleness note below — a stale SDK silently costs ~40 % prefill.

## Setup (Windows)

### Build from source

1. Visual Studio 2022/18 **BuildTools** with the C++ workload (provides CMake + Ninja).
2. Fetch a current TheRock Windows `gfx120X` tarball and extract it:

   ```bat
   curl -s https://nightly.repo.amd.com/rocm/core/tarball/ -o index.html
   rem pick the newest therock-dist-windows-gfx120X-all-<ver>.tar.gz from the embedded file list, then:
   tar -xzf therock-dist-windows-gfx120X-all-<ver>.tar.gz -C C:\path\to\rocm --strip-components=1
   ```

   Check `rocm\lib\llvm\bin\clang++.exe --version` — a stale SDK silently costs ~40 % prefill.
3. Configure and build (Developer Command Prompt, paths adapted):

   ```bat
   cmake -S <source-dir> -B build -G Ninja ^
     -DCMAKE_MAKE_PROGRAM "<vs>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" ^
     -DCMAKE_C_COMPILER "<rocm>\lib\llvm\bin\clang.exe" ^
     -DCMAKE_CXX_COMPILER "<rocm>\lib\llvm\bin\clang++.exe" ^
     -DCMAKE_RC_COMPILER "C:\Program Files (x86)\Windows Kits\10\bin\<ver>\x64\rc.exe" ^
     -DCMAKE_HIP_FLAGS "--rocm-path=<rocm>" ^
     -DCMAKE_PREFIX_PATH "<rocm>" ^
     -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1200 -DBUILD_SHARED_LIBS=ON ^
     -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_HSA_COPY_ENGINE=OFF ^
     -DGGML_OPENMP=OFF -DGGML_RPC=ON -DGGML_NATIVE=OFF -DGGML_STATIC=OFF ^
     -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON ^
     -DGGML_CUDA_FA_QUANTS="bf16-bf16;f16-f16;q8_0-q8_0;q8_0-q5_1;q8_0-q5_0;q8_0-q4_1;q8_0-q4_0;q5_1-q5_1;q5_1-q5_0;q5_1-q4_1;q5_1-q4_0;q5_0-q5_0;q5_0-q4_1;q5_0-q4_0;q4_1-q4_1;q4_0-q4_0"
   cmake --build build -j 8
   ```

   The shipped FA quant list is the 16 pairs above — every K/V combination with K-width >= V-width
   across bf16, f16, q8_0, q5_1, q5_0, q4_1 and q4_0, same coverage as the release binaries. Trim
   `GGML_CUDA_FA_QUANTS` to the pairs you serve to cut build times; every pair needs a matching
   `fattn-vec-instance` file to exist, and a dropped pair falls back to f16 conversion at runtime
   with a one-time warning.

   Do **not** pass `-DCMAKE_SYSTEM_NAME=Windows` / `-DCMAKE_CROSSCOMPILING=ON` for a native build:
   with Ninja they leave `CMAKE_SYSTEM_PROCESSOR` empty, ggml's arch detection fails, and the CPU
   backend silently falls back to `GGML_CPU_GENERIC` (SSE2 only, all `GGML_SSE42/AVX/AVX2/FMA/F16C`
   options ignored). The x86 flags above matter for `--no-mmproj-offload`: the vision encoder then
   runs on `ggml-cpu`, and measured on the Qwen3.8-27B F16 mmproj, generic built 46 s per image vs
   2.5 s with AVX2/FMA/F16C.

4. Run the binaries with `<rocm>\bin` on `PATH`, or copy the runtime DLLs beside them.

## Quick start: `llama-server`

The serving configuration verified on this build (27B GDN model + DFlash2 draft, streamed q8_0/q5_1 cache, 16 GB card):

```bat
llama-server.exe ^
  -m "C:\path\to\model.gguf" ^
  --alias my-model ^
  -ngl 99 ^
  -c 65536 ^
  -ctk q8_0 ^
  -ctv q5_1 ^
  -b 1024 ^
  -ub 512 ^
  -kvu ^
  -np 1 ^
  -fa on ^
  --kv-stream ^
  --kv-resident-mib 448 ^
  --kv-resident-mib-decode 480 ^
  --prio 2 ^
  --prio-batch 2 ^
  --poll-batch 1 ^
  --load-mode dio ^
  --jinja ^
  --model-draft "C:\path\to\Qwen3.8-27B-DFlash2-Q4_K_M.gguf" ^
  --spec-type draft-dflash,ngram-mod ^
  -ngld all ^
  --spec-draft-n-max 4 ^
  --reasoning on ^
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0 --repeat-penalty 1.0 ^
  --host 127.0.0.1 --port 6767
```

The `--kv-resident-mib` values size the resident KV window to the card's free VRAM; tune them (and `-c`) to your model and workload. Context beyond the resident window streams over PCIe — that is the point of `--kv-stream`.

General llama.cpp documentation (server API, tools, model docs) lives upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

## Credits

- [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm) — the KV-stream line this tree merges: adaptive KV streaming, the RDNA4 paged attention kernels, MTP/DFlash2 speculative decoding.
- [RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) — the original adaptive KV streaming work the achyudh line builds on.
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) — the rdna-boosts patch series (v16) merged here: WMMA flash attention, chunked gated-delta-net prefill, k-quant decode boosts.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the base project.
- [ROCm/TheRock](https://github.com/ROCm/TheRock) — the open build/release system behind the Windows ROCm toolchain used here.

## License

MIT, same as llama.cpp. All upstream copyright and license notices are retained in the source tree and history.
