#include <iostream>
#include <chrono>
#include <cuda.h>
#include <bitset>

#include "tasks/task.h"
#include "common/maskElement.h"
#include "jobs/kernels/vectorAdd.h"
#include "libsmctrl.h"

int main(int argc, char** argv)
{
    cudaFree(0);  // Force CUDA context creation
    uint64_t mask1 = 0b1000000000000000000000000000000000000000000000000000000000000000;
    mask1 = 0b0000000000000000000000000010000000000000000000000000000000000000;

    for (int xxx = 0 ; xxx < 12; xxx++, mask1 >>= 1) {
        for (int jj = 0; jj < 5; jj++) {
            // --- parse args ---
            int vectorSize = 10000000;
            int neededTPCs1 = 1;
            int neededTPCs2 = 1;
            int threadsPerBlock = 128;

            if (argc >= 2) vectorSize = std::atoi(argv[1]);
            if (argc >= 3) neededTPCs1 = std::bitset<64>(std::string(argv[2])).to_ullong();
            if (argc >= 4) neededTPCs2 = std::bitset<64>(std::string(argv[3])).to_ullong();
            if (argc >= 5) threadsPerBlock = std::atoi(argv[4]);

/*        std::cout << "Configuration:\n"
                  << "  vectorSize        = " << vectorSize << "\n"
                  << "  threadsPerBlock   = " << threadsPerBlock << "\n"
                  << "  neededTPCsStream1 = " << neededTPCs1 << "\n"
                  << "  neededTPCsStream2 = " << neededTPCs2 << "\n";*/

            size_t bytes = vectorSize * sizeof(float);

            // --- host allocations & init ---
            float *h_A, *h_B, *h_ref, *h_C1, *h_C2;
            cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault);
            cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault);
            cudaHostAlloc(&h_C1, bytes, cudaHostAllocDefault);
            cudaHostAlloc(&h_C2, bytes, cudaHostAllocDefault);
            h_ref = (float *) malloc(bytes);

            for (int i = 0; i < vectorSize; ++i) {
                h_A[i] = float(i);
                h_B[i] = float(i + i);
                h_ref[i] = h_A[i] + h_B[i];
            }

            // --- CPU timing ---
            //auto cpu_start = std::chrono::high_resolution_clock::now();
            //// compute reference
            //for (int i = 0; i < vectorSize; ++i) {
            //    h_ref[i] = h_A[i] + h_B[i];
            //}
            //auto cpu_end = std::chrono::high_resolution_clock::now();
            //double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
            //std::cout << "CPU computation time: " << cpu_ms << " ms\n";

            // --- device allocations ---
            float *d_A1, *d_B1, *d_C1;
            cudaMalloc(&d_A1, bytes);
            cudaMalloc(&d_B1, bytes);
            cudaMalloc(&d_C1, bytes);
            float *d_A2, *d_B2, *d_C2;
            cudaMalloc(&d_A2, bytes);
            cudaMalloc(&d_B2, bytes);
            cudaMalloc(&d_C2, bytes);

            // copy inputs once
            cudaMemcpy(d_A1, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B1, h_B, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_A2, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B2, h_B, bytes, cudaMemcpyHostToDevice);

            // --- query total TPCs & build masks ---
            int cudaDev = 0;
            cudaGetDevice(&cudaDev);
            uint32_t totalTpcs = 0;
            if (libsmctrl_get_tpc_info_cuda(&totalTpcs, cudaDev) != 0) {
                std::cerr << "Failed to get TPC info" << std::endl;
                return 1;
            }
            auto makeDisableMask = [&](int keep) {
                uint64_t en = (keep >= (int) totalTpcs)
                              ? ~0ULL
                              : ((1ULL << keep) - 1ULL);
                return ~en;
            };
            // uint64_t mask1 = makeDisableMask(neededTPCs1);
            // uint64_t mask2 = makeDisableMask(neededTPCs2) << neededTPCs1;
            // mask2 += 2;


            // --- create streams, events and apply masks ---
            cudaStream_t stream1, stream2;
            cudaEvent_t start1, end1, start2, end2;
            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
            cudaEventCreate(&start1);
            cudaEventCreate(&end1);
            cudaEventCreate(&start2);
            cudaEventCreate(&end2);

            //uint64_t mask1 = ~0b10;
            const uint64_t actualMask1 = ~mask1;

            uint64_t mask2 = ~0b01;
            std::cout << std::bitset<64>(actualMask1) << std::endl;
            // std::cout << std::bitset<64>(mask2) << std::endl;


            libsmctrl_set_stream_mask((void *) stream1, actualMask1);
            // libsmctrl_set_stream_mask((void*)stream2, mask2);

            int blocks = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
            // cudaEventRecord(start2, stream2);
            // vectorAddKernel<<<blocks, threadsPerBlock, 0, stream2>>>(d_A2, d_B2, d_C2, vectorSize);


            // --- launch first kernel and time ---
            cudaEventRecord(start1, stream1);
            vectorAddKernel<<<blocks, threadsPerBlock, 0, stream1>>>(d_A1, d_B1, d_C1, vectorSize);

            // --- launch second kernel and time ---
            cudaEventRecord(end1, stream1);
            // cudaEventRecord(end2, stream2);
            // copy results back
            cudaMemcpyAsync(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost, stream1);
            // cudaMemcpyAsync(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost, stream2);

            cudaStreamSynchronize(stream1);
            // cudaStreamSynchronize(stream2);

            float gpu_ms1 = 0.0f, gpu_ms2 = 0.0f;
            cudaEventElapsedTime(&gpu_ms1, start1, end1);
            // cudaEventElapsedTime(&gpu_ms2, start2, end2);

            std::cout << "GPU kernel1 time: " << gpu_ms1 << " ms\n";
            // std::cout << "GPU kernel2 time: " << gpu_ms2 << " ms\n";

            // --- verify results ---
            bool ok1 = true, ok2 = true;
            for (int i = 0; i < vectorSize; ++i) {
                if (fabs(h_C1[i] - h_ref[i]) > 1e-5f) {
                    ok1 = false;
                    break;
                }
                if (fabs(h_C2[i] - h_ref[i]) > 1e-5f) {
                    ok2 = false;
                    break;
                }
            }
            std::cout << "Stream1 result: " << (ok1 ? "PASS" : "FAIL") << "\n";
            // std::cout << "Stream2 result: " << (ok2 ? "PASS" : "FAIL") << "\n";

            // --- cleanup ---
            cudaEventDestroy(start1);
            cudaEventDestroy(end1);
            cudaEventDestroy(start2);
            cudaEventDestroy(end2);
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
            cudaFree(d_A1);
            cudaFree(d_B1);
            cudaFree(d_C1);
            cudaFree(d_A2);
            cudaFree(d_B2);
            cudaFree(d_C2);
            cudaFreeHost(h_A);
            cudaFreeHost(h_B);
            cudaFreeHost(h_C1);
            cudaFreeHost(h_C2);
            free(h_ref);
        }
    }
    //return (ok1 && ok2) ? 0 : 1;
}

/*
1111111111111111111111111111111111111111111111111111111111111110 1111111111111111111111111111111111111111111111111111111111110001

make && ./WCET_evaluation 1000000 1111111111111111111111111111111111111111111111111111111111111110 1111111111111111111111111111111111111111111111111111111111110001
make && ./WCET_evaluation 1000000 1111111111111111111111111111111111111111111111111111111111110001 1111111111111111111111111111111111111111111111111111111111111110
*/
