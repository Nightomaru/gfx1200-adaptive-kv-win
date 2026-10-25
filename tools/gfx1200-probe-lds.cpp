// Print the device limits that decide llama.cpp's MMQ tile width (smpbo) and the HIP runtime
// build, so the same binary can be compared against two different ROCm runtimes.
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int n = 0;
    hipGetDeviceCount(&n);
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t pr;
        if (hipGetDeviceProperties(&pr, i) != hipSuccess) continue;
        int shmem_block = 0, shmem_mp = 0, regs = 0;
        hipDeviceGetAttribute(&shmem_block, hipDeviceAttributeMaxSharedMemoryPerBlock, i);
        hipDeviceGetAttribute(&shmem_mp, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, i);
        hipDeviceGetAttribute(&regs, hipDeviceAttributeMaxRegistersPerMultiprocessor, i);
        printf("dev %d %-24s arch=%-9s sharedMemPerBlock=%-7zu sharedMemPerMultiprocessor=%-7zu "
               "sharedMemPerBlockOptin=%-7zu regsPerMultiProcessor=%d\n",
               i, pr.name, pr.gcnArchName, pr.sharedMemPerBlock, pr.sharedMemPerMultiprocessor,
               pr.sharedMemPerBlockOptin, regs);
    }
    printf("hip runtime: %d.%d (%s)\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR, hipGetErrorName(hipSuccess));
    return 0;
}
