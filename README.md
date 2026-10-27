# llama.cpp (gfx1200-adaptive-kv-win)

**Adaptive KV Streaming on native Windows for AMD RDNA 4 — built and tuned for the Radeon RX 9060 XT (gfx1200, 16 GB).**

This fork takes [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm) branch `feature/adaptive-kv-stream-rocm` — adaptive KV streaming plus RDNA 4 ROCm optimizations, developed on the RX 9070 XT (`gfx1201`) for Linux — and makes it run as a **native Windows** build on a **gfx1200** part, on a current [TheRock](https://github.com/ROCm/TheRock) ROCm, with **DFlash2 draft decoding** and a streamed **q8_0 K / q5_1 V** cache. The result: a 16 GB card runs ~27B-class hybrid models at 128K+ context with the KV cache paged over PCIe, at full attention quality.

---

## What this fork changes (on top of upstream `feature/adaptive-kv-stream-rocm`)

- **Native Windows port of the KV-stream stack.** The upstream copy engine assumes the Linux HSA runtime, which AMD does not ship on Windows. Ported to a HIP stream+event copy path behind `GGML_HIP_HSA_COPY_ENGINE`, after verifying that `hipHostMalloc` is identity-mapped and host-write-coherent on gfx1200/WDDM — so the host-pointer page design works without modification.
- **gfx1200 as a first-class target.** Built for `GPU_TARGETS=gfx1200` (32 CU / 16 WGP); the RDNA4 launch-size heuristics written for the 64-CU 9070 XT were re-measured on the smaller part.
- **A current TheRock ROCm.** Built with `therock-dist-windows-gfx120X-all-10.3.0a20261025` (AMD clang 24).
- **DFlash2 speculative draft.** `--spec-type draft-dflash,ngram-mod` with a separate draft GGUF (`-md` / `--model-draft`).
- **q8_0 K / q5_1 V KV streaming.** New paged decode *and* prefill kernel variants for Q5_1's 24-byte block (5-bit unpack plus per-block offset), selected at compile time so the q4_0 path compiles unchanged.

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
     -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0;q5_0-q5_0;q5_1-q5_1;q8_0-q5_0;q5_0-q5_1;q4_0-q5_0;q8_0-q5_1"
   cmake --build build -j 8
   ```

   Do **not** pass `-DCMAKE_SYSTEM_NAME=Windows` / `-DCMAKE_CROSSCOMPILING=ON` for a native build:
   with Ninja they leave `CMAKE_SYSTEM_PROCESSOR` empty, ggml's arch detection fails, and the CPU
   backend silently falls back to `GGML_CPU_GENERIC` (SSE2 only, all `GGML_SSE42/AVX/AVX2/FMA/F16C`
   options ignored). The x86 flags above matter for `--no-mmproj-offload`: the vision encoder then
   runs on `ggml-cpu`, and measured on the Qwen3.8-27B F16 mmproj, generic built 46 s per image vs
   2.5 s with AVX2/FMA/F16C (Zen 5, 12 threads).

4. Run the binaries with `<rocm>\bin` on `PATH`, or copy the runtime DLLs beside them.

## Quick start: `llama-server`

A serving configuration that ran clean at 128K on the 16 GB card:

```bat
llama-server.exe ^
  -m "C:\path\to\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ^
  --alias Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S ^
  -ngl 99 ^
  -c 131072 ^
  -ctk q8_0 ^
  -ctv q5_1 ^
  -b 1024 ^
  -ub 512 ^
  -kvu ^
  -np 1 ^
  -fa on ^
  --kv-stream ^
  --kv-resident-mib 256 ^
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
  --spec-rs-replay ^
  --reasoning on ^
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0 --repeat-penalty 1.0 ^
  --host 127.0.0.1 --port 6767
```

## Credits

- [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm) — the upstream of this fork: adaptive KV streaming, the RDNA4 paged attention kernels, MTP with rejection sampling. Everything above stands on that work.
- [ROCm/TheRock](https://github.com/ROCm/TheRock) — the open build/release system behind the Windows ROCm toolchain used here.
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) — RDNA performance work and a reference for what the Windows port had to preserve.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the base project.

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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

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
| [CANN](docs/build.md#cann) | Ascend NPU |
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
