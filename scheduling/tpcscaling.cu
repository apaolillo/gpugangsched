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

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

float run_vector_add_with_mask(cudaStream_t stream,
                               const float* a, const float* b, float* c,
                               const std::vector<float>& h_a, const std::vector<float>& h_b,
                               int N, uint64_t allowedMask, int repeat = 5) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    libsmctrl_set_stream_mask((void*) stream, ~allowedMask);

    // Warmup
    vector_add<<<blocks, threadsPerBlock, 0, stream>>>(a, b, c, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times;
    std::vector<float> h_c_result(N);

    for (int i = 0; i < repeat; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        vector_add<<<blocks, threadsPerBlock, 0, stream>>>(a, b, c, N);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        times.push_back(elapsed);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Validate results
        CUDA_CHECK(cudaMemcpy(h_c_result.data(), c, N * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int j = 0; j < N; ++j) {
            float expected = h_a[j] + h_b[j];
            if (std::abs(h_c_result[j] - expected) > 1e-5f) {
                if (errors < 3) {
                    printf("❌ TPC Mask 0x%012lx: error at %d: got %f, expected %f\n",
                           allowedMask, j, h_c_result[j], expected);
                }
                ++errors;
            }
        }

        if (errors > 0) {
            printf("❌ TPC Mask 0x%012lx: %d mismatches on iteration %d — skipping further repeats\n",
                   allowedMask, errors, i);
            exit(1);
        }
    }

    float sum = 0.0f, sq_sum = 0.0f;
    for (float t : times) {
        sum += t;
        sq_sum += t * t;
    }
    float mean = sum / times.size();
    return mean;
}

int main() {
    const int maxTPCs = 12;       // Max TPCs to test
    const int N = 1 << 20;        // Number of elements

    // Allocate and initialize host vectors
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>((i + 1) % 1000) * 0.002f;
    }

    // Allocate device vectors
    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float baseline_time = 0.0f;

    printf("TPCs | Mask         | Time (ms) | Speedup\n");
    printf("-----------------------------------------\n");

    for (int tpcCount = 1; tpcCount <= maxTPCs; ++tpcCount) {
        //uint64_t allowedMask = (1ULL << tpcCount) - 1;
        uint64_t start = (1ULL << 11);
        uint64_t allowedMask = start;
        for (int i = 1; i < tpcCount; i++)
        {
            allowedMask >>= 1;
            allowedMask += start;
        }

        float time = run_vector_add_with_mask(stream, a, b, c, h_a, h_b, N, allowedMask, 5);
        if (tpcCount == 1) baseline_time = time;
        float speedup = baseline_time / time;
        printf("  %2d  | 0x%012lx | %8.3f | %7.2fx\n", tpcCount, allowedMask, time, speedup);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return 0;
}




/*
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "libsmctrl.h"
#include <cmath>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

float run_vector_add_with_mask(cudaStream_t stream, const float* a, const float* b, float* c,
                               int N, uint64_t allowedMask, int repeat = 5) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    libsmctrl_set_stream_mask((void*) stream, ~allowedMask);

    // Warmup
    vector_add<<<blocks, threadsPerBlock, 0, stream>>>(a, b, c, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times;

    for (int i = 0; i < repeat; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        vector_add<<<blocks, threadsPerBlock, 0, stream>>>(a, b, c, N);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        times.push_back(elapsed);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    float sum = 0.0f, sq_sum = 0.0f;
    for (float t : times) {
        sum += t;
        sq_sum += t * t;
    }
    float mean = sum / repeat;
    float stddev = std::sqrt((sq_sum / repeat) - (mean * mean));
    return mean;
}

int main() {
    const int maxTPCs = 12;
    const int N = 1 << 20;

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, N * sizeof(float)));

    // Optionally: initialize `a`, `b` here on host and cudaMemcpy to device

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float baseline_time = 0.0f;

    printf("TPCs | Mask         | Time (ms) | Speedup\n");
    printf("-----------------------------------------\n");

    for (int tpcCount = 1; tpcCount <= maxTPCs; ++tpcCount) {
        uint64_t allowedMask = (1ULL << tpcCount) - 1;
        float time = run_vector_add_with_mask(stream, a, b, c, N, allowedMask, 5);
        if (tpcCount == 1) baseline_time = time;
        float speedup = baseline_time / time;
        printf("  %2d  | 0x%012lx | %8.3f | %7.2fx\n", tpcCount, allowedMask, time, speedup);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return 0;
}




/*
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "libsmctrl.h"
#include <cmath>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void dummy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        for (int i = 0; i < 1000; ++i) {
            x = x * 1.00001f + 0.0001f;
        }
        data[idx] = x;
    }
}

float run_kernel_with_mask(cudaStream_t stream, float* dev_data, int N, uint64_t allowedMask, int repeat = 5) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // In libsmctrl: 0 = allow, 1 = disable → inverse mask logic
    libsmctrl_set_stream_mask((void*) stream, ~allowedMask);

    // Warmup run (not measured)
    dummy_kernel<<<blocks, threadsPerBlock, 0, stream>>>(dev_data, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times;

    for (int i = 0; i < repeat; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        dummy_kernel<<<blocks, threadsPerBlock, 0, stream>>>(dev_data, N);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        times.push_back(elapsed);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Compute average
    float sum = 0.0f, sq_sum = 0.0f;
    for (float t : times) {
        sum += t;
        sq_sum += t * t;
    }
    float mean = sum / (float) repeat;
    float stddev = std::sqrt((sq_sum / repeat) - (mean * mean));

//    printf("Mask: 0x%012lx | Time: %.3f ms ± %.3f ms\n", allowedMask, mean, stddev);
    return mean;
}

int main() {
    const int maxTPCs = 12;
    const int N = 1 << 20;

    float* dev_data;
    CUDA_CHECK(cudaMalloc(&dev_data, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int cudaDev = 0;
    CUDA_CHECK(cudaGetDevice(&cudaDev));

    float baseline_time = 0.0f;

//    const float warmup_time = run_kernel_with_mask(stream, dev_data, N, -1, 5);
//    printf("Warmup time: %.3f ms\n", warmup_time);

    printf("TPCs | Mask         | Time (ms) | Speedup\n");
    printf("-----------------------------------------\n");

    for (int tpcCount = 1; tpcCount <= maxTPCs; ++tpcCount) {
        // Build mask with lowest `tpcCount` bits set to 1 (enabled)
        uint64_t allowedMask = (1ULL << tpcCount) - 1;

        float time = run_kernel_with_mask(stream, dev_data, N, allowedMask, 5);
        if (tpcCount == 1) {
            baseline_time = time;
        }
        float speedup = baseline_time / time;
        printf("  %2d  | 0x%012lx | %8.3f | %7.2fx\n", tpcCount, allowedMask, time, speedup);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dev_data));
    return 0;
}
*/
