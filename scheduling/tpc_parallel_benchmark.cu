#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include "libsmctrl.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Simple vector add kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Per-kernel configuration
struct KernelConfig {
    float* a;
    float* b;
    float* c;
    uint64_t tpc_mask;
    cudaStream_t stream;
    float time_ms;
};

// Launch multiple vector_add kernels concurrently with custom TPC masks
void run_concurrent_vector_add(std::vector<KernelConfig>& kernels, int N) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<cudaEvent_t> starts(kernels.size()), stops(kernels.size());

    for (size_t i = 0; i < kernels.size(); ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));

        // Set allowed TPCs (libsmctrl inverts the mask)
        libsmctrl_set_stream_mask((void*) kernels[i].stream, ~kernels[i].tpc_mask);

        CUDA_CHECK(cudaEventRecord(starts[i], kernels[i].stream));
        vector_add<<<blocks, threadsPerBlock, 0, kernels[i].stream>>>(
            kernels[i].a, kernels[i].b, kernels[i].c, N);
        CUDA_CHECK(cudaEventRecord(stops[i], kernels[i].stream));
    }

    for (size_t i = 0; i < kernels.size(); ++i) {
        CUDA_CHECK(cudaEventSynchronize(stops[i]));
        CUDA_CHECK(cudaEventElapsedTime(&kernels[i].time_ms, starts[i], stops[i]));
        CUDA_CHECK(cudaEventDestroy(starts[i]));
        CUDA_CHECK(cudaEventDestroy(stops[i]));
    }
}

int main() {
    // Need to run these two to set the driver persistence & frequency:
    //   sudo nvidia-smi -pm 1
    //   sudo nvidia-smi -lgc 2115

    const int N = 1 << 20;  // 4MB per vector
    const int num_kernels = 2;
    std::vector<double> avgs(num_kernels);
    const size_t rep = 1000;
    uint64_t mask0 = 0b00000011;
    uint64_t mask1 = 0b10000000;

    for (size_t ii = 0; ii <= rep; ii++) {

        std::vector<KernelConfig> kernels(num_kernels);

        for (int i = 0; i < num_kernels; ++i) {
            CUDA_CHECK(cudaMalloc(&kernels[i].a, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&kernels[i].b, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&kernels[i].c, N * sizeof(float)));
            CUDA_CHECK(cudaStreamCreate(&kernels[i].stream));

            // Configure TPC masks: overlap or isolate
            kernels[i].tpc_mask = (i == 0) ? mask0 : mask1;
        }

        run_concurrent_vector_add(kernels, N);

        printf("\nResults run %zu:\n", ii);
        for (int i = 0; i < num_kernels; ++i) {
            if (ii > 10) {
                avgs[i] += kernels[i].time_ms;
            }
            printf("Kernel %d | Mask: 0x%012lx | Time: %.3f ms\n",
                   i, kernels[i].tpc_mask, kernels[i].time_ms);
        }

        // Cleanup
        for (int i = 0; i < num_kernels; ++i) {
            CUDA_CHECK(cudaFree(kernels[i].a));
            CUDA_CHECK(cudaFree(kernels[i].b));
            CUDA_CHECK(cudaFree(kernels[i].c));
            CUDA_CHECK(cudaStreamDestroy(kernels[i].stream));
        }
    }

    printf("\n\nAverages over %zu runs for %d parallel kernels:\n", rep, num_kernels);
    for (int i = 0; i < num_kernels; ++i) {
        const uint64_t mask = (i == 0) ? mask0 : mask1;
        printf("Kernel %d | Mask: 0x%012lx | Avg Time: %.3f ms\n",
               i, mask, avgs[i] / rep);
    }

    return 0;
}
