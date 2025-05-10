#include <cstdio>
#include <cuda_runtime.h>
#include "libsmctrl.h"

__device__ unsigned get_smid() {
    unsigned smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

__global__ void log_smid(int* smid_log) {
    if (threadIdx.x == 0) {  // One thread per block logs the SM ID
        int smid = get_smid();
        smid_log[blockIdx.x] = smid;
    }
}

int main() {
    const int numBlocks = 64;  // Try with a value >= number of SMs
    const int threadsPerBlock = 32;

    int* smid_log;
    int* smid_log_host = new int[numBlocks];

    int cudaDev = 0;
    cudaGetDevice(&cudaDev);
    uint32_t totalTpcs = 0;
    if (libsmctrl_get_tpc_info_cuda(&totalTpcs, cudaDev) != 0) {
        return 1;
    }

    libsmctrl_set_stream_mask((void *) stream1, actualMask1);

    cudaMalloc(&smid_log, numBlocks * sizeof(int));
    log_smid<<<numBlocks, threadsPerBlock>>>(smid_log);
    cudaMemcpy(smid_log_host, smid_log, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Block -> SMID mapping:\n");
    for (int i = 0; i < numBlocks; ++i) {
        printf("Block %2d ran on SM %2d\n", i, smid_log_host[i]);
    }

    cudaFree(smid_log);
    delete[] smid_log_host;
    return 0;
}
