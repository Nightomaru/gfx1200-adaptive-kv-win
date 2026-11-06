# llama.cpp (gfx1200-kv-resident-win)

**Windowed KV streaming + RDNA boosts on native Windows for AMD RDNA 4 (gfx1200 / RX 9060 XT).**

This fork is a current llama.cpp with the full [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) delivery (`v16-6050b467c-r220`, 16 blocks, strict `git am`, tree-verified against the canonical delivery) and the rdna-boosts **resident-window KV** (block 15) made to run on **native Windows**: the newest `N` KV cells stay in VRAM, the older tail spills to pinned system RAM inside one VA reservation — numerically transparent, no kernel changes, so a 16 GB card serves 27B-class models at 100K+ context.

## What this build does

* **Newer llama.cpp** — upstream `6050b467c` (2026-10-25) + all 16 rdna-boosts blocks (FA band + native quantized K/V, chunked gated-delta-net prefill, MTP draft depth, k-quant boosts, fused MoE, host-resident MoE, slab + async input, qwen4exp, streamed KV).
* **`--kv-resident-cells N|auto`** — enables the resident-window KV: N cells resident in VRAM, the rest spilled to pinned RAM (`auto` + `--fit` sizes it from your hardware; `0`/unset = fully resident, byte-identical to stock).
* **RDNA boost for Windows, targeting `gfx1200`** — the full boost set compiled for RDNA4 on a Windows TheRock ROCm toolchain, plus Windows-specific fixes:
  * the region provider **probes** host-location VMM at runtime: driver stacks that support it run the window, stacks that do not decline cleanly with one message (never a hang or a crash);
  * an enumerated iGPU peer is never granted VMM access (`hipMemSetAccess` faults the process on some driver stacks);
  * `origami.dll` (hipBLASLt/TensileLite runtime dependency) is documented as part of a redistributable bundle.

Two platform notes that matter:

* **Do not put the build SDK's `bin` folder on `PATH` when serving.** The SDK-bundled `amdocl64.dll` lacks host-location VMM support; the Windows driver-store `amdocl64` (used automatically when not shadowed) has it. The capability probe handles it either way, but the window only *works* with the driver-store stack.
* GSQ/`qwen35`-family models require **uniform K/V cache types** on this tree (`-ctk q8_0 -ctv q8_0`); mixed pairs like `q8_0`/`q5_1` are rejected.

## Building on Windows

1. **Visual Studio 2022/18 Build Tools** with the C++ workload (provides CMake + Ninja).
2. **TheRock Windows `gfx120X` tarball** (ROCm toolchain + runtime):

   ```
   curl -O https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx120X-all-10.3.0a20261026.tar.gz
   tar -xzf therock-dist-windows-gfx120X-all-10.3.0a20261026.tar.gz -C C:\path\to\rocm
   ```

   Use **`10.3.0a20261026` (b1356)** — newer nightly builds measured ~40 % lower prefill on RDNA4. Do **not** add `<rocm>\bin` to `PATH` for serving (see the note above).

3. **Configure and build** from a Developer Command Prompt (native configure — do **not** set `CMAKE_SYSTEM_NAME`/`CMAKE_CROSSCOMPILING`, that silently builds ggml-cpu as SSE2-only):

   ```
   cmake -S <source-dir> -B build -G Ninja ^
     -DCMAKE_MAKE_PROGRAM "<vs>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" ^
     -DCMAKE_C_COMPILER "<rocm>\lib\llvm\bin\clang.exe" ^
     -DCMAKE_CXX_COMPILER "<rocm>\lib\llvm\bin\clang++.exe" ^
     -DCMAKE_RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" ^
     -DCMAKE_HIP_FLAGS "--rocm-path=<rocm>" ^
     -DCMAKE_PREFIX_PATH "<rocm>" ^
     -DCMAKE_BUILD_TYPE=Release ^
     -DGPU_TARGETS=gfx1200 ^
     -DBUILD_SHARED_LIBS=ON ^
     -DLLAMA_BUILD_SERVER=ON ^
     -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_HSA_COPY_ENGINE=OFF ^
     -DGGML_OPENMP=OFF -DGGML_RPC=ON -DGGML_NATIVE=OFF -DGGML_STATIC=OFF ^
     -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON ^
     -DGGML_CUDA_FA_QUANTS="bf16-bf16;f16-f16;q8_0-q8_0;q8_0-q5_1;q8_0-q5_0;q8_0-q4_1;q8_0-q4_0;q5_1-q5_1;q5_1-q5_0;q5_1-q4_1;q5_1-q4_0;q5_0-q5_0;q5_0-q4_1;q5_0-q4_0;q4_1-q4_1;q4_0-q4_0"
   cmake --build build -j 12
   ```

4. Run the binaries with the driver's ROCm stack (default), or bundle the runtime DLLs from `<rocm>\bin` **plus `origami.dll`** beside them. Do not bundle/PATH the SDK's `amdocl64.dll`.

## Quick start: `llama-server` at 102K on a 16 GB card

A serving configuration exercised at 102400 context with the window sized by `--fit`:

```
llama-server.exe ^
  -m "C:\path\to\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ^
  --alias Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S ^
  -ngl 99 ^
  -t 12 -tb 8 ^
  -c 102400 ^
  -ctk q8_0 -ctv q8_0 ^
  -b 1024 -ub 512 ^
  --kv-resident-cells auto ^
  -kvu ^
  -np 1 ^
  -fa on ^
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
  --reasoning-preserve ^
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0 --repeat-penalty 1.0 ^
  --host 127.0.0.1 --port 6767
```

Drop the `--model-draft` / `--spec-type` / `-ngld` lines for target-only serving. Related knobs: `LLAMA_KV_SPILL_RAM_PCT` (pinned-RAM ceiling for the spill, default 85 % of available RAM), and `--fit` prints the resident/spill split it chose. `HIP_VISIBLE_DEVICES=0` is worth setting on systems with an enumerated iGPU.

## Credits

* [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) — the RDNA boost patch set this build applies (blocks 00-15 of `v16-6050b467c-r220`), including the streamed-KV resident window this fork makes runnable on Windows.
* [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm) — the original adaptive-KV-streaming ROCm fork that demonstrated KV streaming on RDNA 4 and inspired this direction.
* [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the base project.
* AMD TheRock — the open Windows ROCm toolchain this build uses.

---

*(Original upstream llama.cpp documentation follows below)*

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

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
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Huawei Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
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
