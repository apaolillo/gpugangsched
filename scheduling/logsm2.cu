#include <cstdio>
#include <cuda_runtime.h>
#include "libsmctrl.h"

__device__ int get_smid() {
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

__global__ void log_smid(int* smid_log) {
    if (threadIdx.x == 0) {
        int smid = get_smid();
        smid_log[blockIdx.x] = smid;
    }
}

int main() {
    const bool use_smctrl = true;
    const uint64_t mask = 0b0000000000000000000000000000000000000000000000000000100000000000;
//    const uint64_t mask = 0b0000000000000000000000000000000000000000000000000000111111111111;
    const int numBlocks = 64;
    const int threadsPerBlock = 32;

    int* smid_log_device;
    int* smid_log_host = new int[numBlocks];

    // Allocate memory
    cudaMalloc(&smid_log_device, numBlocks * sizeof(int));

    // Create a stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int cudaDev = 0;
    cudaGetDevice(&cudaDev);
    uint32_t totalTpcs = 0;
    if (libsmctrl_get_tpc_info_cuda(&totalTpcs, cudaDev) != 0) {
        return 1;
    }
    const uint64_t actualMask = ~mask;
    if (use_smctrl) {
        libsmctrl_set_stream_mask((void *) stream, actualMask);
    }

    // Launch kernel in the created stream
    log_smid<<<numBlocks, threadsPerBlock, 0, stream>>>(smid_log_device);

    // Async copy from device to host in the same stream
    cudaMemcpyAsync(smid_log_host, smid_log_device, numBlocks * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    // Wait for all work in the stream to complete
    cudaStreamSynchronize(stream);

    // Print result
    printf("Block -> SMID mapping (via stream):\n");
    for (int i = 0; i < numBlocks; ++i) {
        printf("Block %2d ran on SM %2d\n", i, smid_log_host[i]);
    }

    // Cleanup
    cudaFree(smid_log_device);
    delete[] smid_log_host;
    cudaStreamDestroy(stream);


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
}
