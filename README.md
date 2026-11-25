# llama.cpp (gfx1200-kv-resident-win)

**Windowed KV streaming + RDNA boosts on native Windows for AMD RDNA 4 (gfx1200 / RX 9060 XT).**

This fork is a current llama.cpp with the full [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) delivery (`v18-919794500-r357`, 18 official blocks + temporary `0300`, strict `git am`, tree-verified against the canonical delivery) and the rdna-boosts **resident-window KV** (block 14) made to run on **native Windows**: the newest `N` KV cells stay in VRAM, the older tail spills to pinned system RAM inside one VA reservation - numerically transparent, so a 16 GB card serves 27B-class models at 100K+ context.

Current build: `b12619` @ `d9816ba20` = upstream `919794500` (2026-11-20, +228 commits past the previous r327 base) + the r357 delivery + the two Windows-specific VMM guard commits and the CI park.

## What this build does

* **Newer llama.cpp** - upstream `919794500` + all 18 rdna-boosts blocks (FA band + native quantized K/V, chunked gated-delta-net prefill, adaptive MTP draft depth, k-quant boosts, fused kernels, host-resident MoE, slab + async input, qwen4exp, the streamed KV, the UMA GTT expert store, the server's prompt admission policy, GPU ternary weights) plus the temporary relaxed-tensor-split block. Native Inkling 2 model support comes with the new base.
* **`--kv-window auto|N`** - enables the resident-window KV (the delivery's first-class option; the old `LLAMA_KV_RESIDENT_CELLS` env var still works as a deprecated alias): N cells resident in VRAM, the rest spilled to pinned RAM (`auto` + `--fit` sizes it from your hardware; unset/0 = fully resident, byte-identical to stock).
* **New in r357:** block 17 adds `TQ1_0`/`TQ2_0` ternary weights on the GPU (`GGML_CUDA_TQ=0` disables them); a block-12 amendment keeps tool-result and tool-call text from forging chat-turn tokens (`LLAMA_SERVER_TOOL_TEXT=0` restores stock). Carried forward from r327: server prompt admission (`-ib`), UMA GTT expert store, RDNA2 kernel amendments and the host-expert routed-MoE ubatch fix.
* **RDNA boost for Windows, targeting `gfx1200`** - the full boost set compiled for RDNA4 on a Windows TheRock ROCm toolchain, plus Windows-specific fixes:
  * the region provider **probes** host-location VMM at runtime: driver stacks that support it run the window, stacks that do not decline cleanly with one message (never a hang or a crash);
  * an enumerated iGPU peer is never granted VMM access (`hipMemSetAccess` faults the process on some driver stacks - measured with a Raphael iGPU enumerated next to the RX 9060 XT; the guard declines every peer grant, zero faults).

Two platform notes that matter:

* **Do not put the build SDK's `bin` folder on `PATH` when serving.** The SDK-bundled `amdocl64.dll` lacks host-location VMM support; the driver-store `amdocl64` (used automatically when not shadowed) has it. The capability probe handles it either way, but the window only *works* with the driver-store stack.
* GSQ/`qwen35`-family models require **uniform K/V cache types** on this tree (`-ctk q8_0 -ctv q8_0`); mixed pairs like `q8_0`/`q5_1` are rejected. (The delivery's optional `mixed-kv` artifact re-enables some mixed pairs for 256-wide-attention models; it is not applied in this build.)

## ROCm SDK

* **TheRock 10.2.0 stable** (`therock-dist-windows-gfx120X-all-10.2.0.tar.gz`, released 2026-11-18) - the build SDK for this release; it ships `libtensilelite-host.dll` separately, so bundle it beside the binaries. **TheRock 10.1.0 stable** remains a validated fallback.
* Do **not** use the `10.3.0a20261026` nightly (`ld.lld` crash on the FA-MMA instances) or `20261031+` nightlies (hipBLASLt logit drift and lower prefill).

## Building on Windows

1. **Visual Studio 2022/18 Build Tools** with the C++ workload (provides CMake + Ninja).
2. **TheRock Windows `gfx120X` tarball** (ROCm toolchain + runtime):

   ```
   curl -O https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx120X-all-10.2.0.tar.gz
   tar -xzf therock-dist-windows-gfx120X-all-10.2.0.tar.gz -C C:\path\to\rocm
   ```

3. **Configure natively** (do **not** set `CMAKE_SYSTEM_NAME`/`CMAKE_CROSSCOMPILING`, that silently builds ggml-cpu as SSE2-only):

   ```
   cmake -S <source-dir> -B build -G Ninja ^
     -DCMAKE_MAKE_PROGRAM "<vs>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" ^
     -DCMAKE_C_COMPILER "<rocm>\lib\llvm\bin\clang.exe" ^
     -DCMAKE_CXX_COMPILER "<rocm>\lib\llvm\bin\clang++.exe" ^
     -DCMAKE_HIP_FLAGS "--rocm-path=<rocm>" ^
     -DCMAKE_PREFIX_PATH "<rocm>" ^
     -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1200 ^
     -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_SERVER=ON ^
     -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_HSA_COPY_ENGINE=OFF ^
     -DGGML_OPENMP=OFF -DGGML_RPC=ON -DGGML_NATIVE=OFF -DGGML_STATIC=OFF
   cmake --build build -j 12
   ```

4. Bundle the runtime DLLs from `<rocm>\bin` **plus `origami.dll`** (and `libtensilelite-host.dll` on 10.2.0) beside the binaries, with the `rocblas/` and `hipblaslt/` data trees. Do not bundle/PATH the SDK's `amdocl64.dll`.

## Quick start: `llama-server` at long context on a 16 GB card

```
llama-server.exe -m path\to\model.gguf -ngl 99 --fit on --jinja ^
  -c 102400 -ctk q8_0 -ctv q8_0 --kv-window auto --host 127.0.0.1 --port 6767
```

Do NOT lower `-ngl` on Windows to make a model fit: the RAM offload OOMs the 16 GB system RAM. Related knobs: `LLAMA_KV_SPILL_RAM_PCT` (pinned-RAM ceiling for the spill, default 85 % of available RAM); `HIP_VISIBLE_DEVICES=0` is worth setting on systems with an enumerated iGPU.

## Credits

* [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) - the RDNA boost patch set (blocks 00-17 + `0300` of `v18-919794500-r357`), including the streamed-KV resident window this fork makes runnable on Windows.
* [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm) - the original adaptive-KV-streaming ROCm fork that demonstrated KV streaming on RDNA 4 and inspired this direction.
* [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - the base project.
* AMD TheRock - the open Windows ROCm toolchain this build uses.

---

*(Original upstream llama.cpp documentation follows below)*

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
