// gfx1200 (RX 9060 XT) probe for the RDNA4 llama.cpp fork port:
//
//  1. device geometry -> does multiProcessorCount count WGPs or CUs? (the fork computes
//     launch sizes as nsm * 64 wave slots, tuned on a 64-CU gfx1201)
//  2. pinned host memory -> is the device VA identical to the host VA on this Windows/WDDM
//     driver? (the fork's KV streaming hands HOST pointers straight to kernels: the sync
//     flags, the page table and the tag are host memory that device kernels dereference)
//  3. host-write -> device-poll visibility (coherency of the KV streaming "ready" flag)
//  4. the launch-size cliff: time a null kernel around candidate wave-slot multiples.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

__global__ void k_null() {}

__global__ void k_read_host(const uint32_t * p, uint32_t * out) {
    if (threadIdx.x == 0) {
        *out = p[0];
    }
}

// polls a host-mapped flag like the paged-decode attention does
__global__ void k_poll(const uint32_t * flag, int64_t limit, int * result) {
    if (threadIdx.x == 0) {
        int64_t i = 0;
        for (; i < limit; ++i) {
            if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) == 1u) {
                break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
        *result = i < limit ? 1 : 0;
    }
}

static float time_null(int blocks, int threads, int iters) {
    hipEvent_t a, b;
    hipEventCreate(&a);
    hipEventCreate(&b);
    // warm up
    for (int i = 0; i < 20; ++i) {
        k_null<<<blocks, threads>>>();
    }
    hipDeviceSynchronize();
    hipEventRecord(a);
    for (int i = 0; i < iters; ++i) {
        k_null<<<blocks, threads>>>();
    }
    hipEventRecord(b);
    hipEventSynchronize(b);
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a);
    hipEventDestroy(b);
    return ms * 1000.0f / iters; // us per launch
}

int main() {
    int n_dev = 0;
    hipGetDeviceCount(&n_dev);
    printf("hipGetDeviceCount = %d\n", n_dev);
    for (int i = 0; i < n_dev; ++i) {
        hipDeviceProp_t pr;
        if (hipGetDeviceProperties(&pr, i) != hipSuccess) continue;
        int mp = 0, mt = 0;
        hipDeviceGetAttribute(&mp, hipDeviceAttributeMultiprocessorCount, i);
        hipDeviceGetAttribute(&mt, hipDeviceAttributeMaxThreadsPerMultiProcessor, i);
        printf("dev %d: %s | arch %s | warp %d | multiProcessorCount %d | maxThreadsPerMP %d | memBusWidth %d | clock %d kHz\n",
               i, pr.name, pr.gcnArchName, pr.warpSize, mp, mt, pr.memoryBusWidth, pr.clockRate);
        printf("        totalGlobalMem %.2f GiB | sharedMemPerBlock %zu | regsPerBlock %d\n",
               pr.totalGlobalMem / 1073741824.0, pr.sharedMemPerBlock, pr.regsPerBlock);
    }

    hipSetDevice(0);

    // ---- pinned host memory: identity mapping? ----
    for (int variant = 0; variant < 3; ++variant) {
        unsigned flags = variant == 0 ? hipHostMallocDefault
                       : variant == 1 ? hipHostMallocMapped
                                      : (hipHostMallocMapped | hipHostMallocCoherent);
        void * h = nullptr;
        hipError_t e = hipHostMalloc(&h, 4096, flags);
        printf("\n[variant %d] hipHostMalloc(flags=0x%x) -> %s host=%p\n", variant, flags, hipGetErrorString(e), h);
        if (e != hipSuccess) {
            hipGetLastError();
            continue;
        }
        uint32_t * hp = (uint32_t *) h;
        uint32_t * dp = nullptr;
        e = hipHostGetDevicePointer((void **) &dp, h, 0);
        printf("            hipHostGetDevicePointer -> %s dev=%p identity=%s\n",
               hipGetErrorString(e), dp, (dp == hp) ? "YES" : "NO");
        if (e != hipSuccess) {
            hipGetLastError();
            hipFreeHost(h);
            continue;
        }

        // device read of host memory
        uint32_t * out = nullptr;
        if (hipMalloc(&out, 4) == hipSuccess) {
            hipMemset(out, 0, 4);
            hp[0] = 0xABCD1234u;
            k_read_host<<<1, 32>>>(dp, out);
            hipError_t ke = hipDeviceSynchronize();
            uint32_t got = 0;
            hipMemcpy(&got, out, 4, hipMemcpyDeviceToHost);
            printf("            device read (kernel printf-free) -> %s value=0x%08x %s\n",
                   hipGetErrorString(ke), got, got == 0xABCD1234u ? "OK" : "MISMATCH");
            hipFree(out);
        }

        // host write -> device poll (the KV streaming 'ready' flag path)
        int * res = nullptr;
        if (hipMalloc(&res, 4) == hipSuccess) {
            int zero = 0;
            hipMemcpy(res, &zero, 4, hipMemcpyHostToDevice);
            hp[0] = 0;
            // do NOT synchronise here: the kernel spins until the host writes the flag
            k_poll<<<1, 32>>>(dp, 40000000LL, res);  // bounded: gives up after ~40M polls
            for (volatile int i = 0; i < 2000000; ++i) {}   // let it start spinning
            __atomic_store_n(hp, 1u, __ATOMIC_RELEASE);     // host write, like the publisher thread
            hipError_t ke = hipDeviceSynchronize();
            int seen = 0;
            hipMemcpy(&seen, res, 4, hipMemcpyDeviceToHost);
            printf("            host write -> device poll -> %s seen=%d %s\n",
                   hipGetErrorString(ke), seen, seen ? "COHERENT" : "NOT SEEN (hang risk)");
            hipFree(res);
        }
        hipFreeHost(h);
    }

    // ---- the launch cliff: gfx1201 was 2048 slots (32 WGPs x 64); what is it here? ----
    printf("\nlaunch-size scan (us per null launch, 128 threads/block):\n");
    const int cand[] = { 254, 255, 256, 257, 258, 510, 511, 512, 513, 514,
                         1022, 1023, 1024, 1025, 1026, 2046, 2047, 2048, 2049, 2050, 4096 };
    for (size_t i = 0; i < sizeof(cand) / sizeof(cand[0]); ++i) {
        const float us = time_null(cand[i], 128, 100);
        printf("   blocks=%-5d  %8.3f us   (waves=%d)\n", cand[i], us, (cand[i] * 4 + 63) / 64);
    }

    printf("\ndone\n");
    return 0;
}
