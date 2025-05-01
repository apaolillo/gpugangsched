#include <iostream>
#include <chrono>
#include <cuda.h>

#include "tasks/task.h"
#include "common/maskElement.h"
#include "jobs/kernels/vectorAdd.h"
#include "libsmctrl.h"

int main(int argc, char** argv)
{
  int vectorSize = 128;
  // if (argc >= 2) {
  //   vectorSize = atoi(argv[1]);
  // }

  int threadsPerBlock = 128;
  int neededTPCs = 1;
  int threadBlocks; // Calculated

  if (argc >= 2) vectorSize = std::atoi(argv[1]);
  if (argc >= 3) neededTPCs = std::atoi(argv[2]);
  if (argc >= 4) threadsPerBlock = std::atoi(argv[3]);

  std::cout << "Configuration:\n";
  std::cout << "vectorSize = " << vectorSize << "\n";
  std::cout << "threadsPerBlock = " << threadsPerBlock << "\n";
  std::cout << "TPCS = " << neededTPCs << "\n";

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;

  cudaStream_t kernelStream;
  std::vector<MaskElement> TPCMasks;

  cudaError_t err;
  err = cudaMalloc(&d_A, vectorSize * sizeof(float));
  err = cudaMalloc(&d_B, vectorSize * sizeof(float));
  err = cudaMalloc(&d_C, vectorSize * sizeof(float));

  err = cudaStreamCreate(&kernelStream);

  int nrOfElements = vectorSize * sizeof(float);
  err = cudaHostAlloc((void **)&A, nrOfElements, cudaHostAllocDefault);
  err = cudaHostAlloc((void **)&B, nrOfElements, cudaHostAllocDefault);
  err = cudaHostAlloc((void **)&C, nrOfElements, cudaHostAllocDefault);

  int totalThreads = vectorSize;
  threadBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

  for (int i = 0; i < vectorSize; i++) {
    A[i] = i;
    B[i] = i + i;
  }

  float *C_res_CPU = (float *)malloc(vectorSize * sizeof(float));
  if (!C_res_CPU) {
    std::cerr << "Failed to allocate C_res_CPU" << std::endl;
    return 1;
  }

  // std::cout << "Computing with CPU..." << std::endl;
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < vectorSize; ++i) {
    C_res_CPU[i] = A[i] + B[i];
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
  std::cout << "CPU computation time: " << cpu_duration / 1000.0 << std::endl;
  
  // libsmctrl Control number of TPC
  int cudaDev; // Could use DeviceInfo singleton of Math
  cudaGetDevice(&cudaDev);
  uint32_t num_tpcs = 0;
  if (libsmctrl_get_tpc_info_cuda(&num_tpcs, cudaDev) != 0) {
      std::cerr << "Failed to get TPC info from libsmctrl" << std::endl;
      return 1;
  }
  std::cout << "Number of TPCs: " << neededTPCs << std::endl;
  if (neededTPCs < 1 || neededTPCs > static_cast<int>(num_tpcs)) {
      std::cerr << "Invalid neededTPCs: " << neededTPCs << " (must be 1.." << num_tpcs << ")" << std::endl;
      return 1;
  }
  // Build mask: enable first `neededTPCs` TPCs (IDs 0..neededTPCs-1)
  uint64_t enableMask = (neededTPCs == 64)
      ? ~0ULL
      : ((1ULL << neededTPCs) - 1ULL);
  // Mask bits = 1 -> disable those TPCs, so invert
  uint64_t disableMask = ~enableMask;
  // Apply mask to our stream (overrides global mask)
  libsmctrl_set_stream_mask((void*)kernelStream, disableMask);

  err = cudaMemcpyAsync(d_A, A, nrOfElements, cudaMemcpyHostToDevice, kernelStream);
  err = cudaMemcpyAsync(d_B, B, nrOfElements, cudaMemcpyHostToDevice, kernelStream);

  cudaEvent_t start, end;
  float elapsedTime;

  // Create CUDA events
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, kernelStream);

  
  // std::cout << "Computing with GPU..." << std::endl;
  auto gpu_start = std::chrono::high_resolution_clock::now();
  
  vectorAddKernel<<<threadBlocks, threadsPerBlock, 0, kernelStream>>>(d_A, d_B, d_C, vectorSize);
  err = cudaGetLastError();

  cudaEventRecord(end, kernelStream);

  // Calculate the execution time
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);

  // Print the execution time
  std::cout << "GPU computation time (without. copy & sync): " << elapsedTime << std::endl;
  
  err = cudaMemcpyAsync(C, d_C, nrOfElements, cudaMemcpyDeviceToHost, kernelStream);
  err = cudaStreamSynchronize(kernelStream);
  
  auto gpu_end = std::chrono::high_resolution_clock::now();
  auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
  std::cout << "GPU computation time (incl. copy & sync): " << gpu_duration / 1000.0 << std::endl;
  std::cout << "GPU synchronization time: " << gpu_duration / 1000.0 - elapsedTime << std::endl;
  
  bool cpu_gpu_match = true;
  for (int i = 0; i < vectorSize; ++i) {
    if (fabs(C_res_CPU[i] - C[i]) > 1e-5) {
      std::cerr << "Mismatch at index " << i << ": CPU = " << C_res_CPU[i] << ", GPU = " << C[i] << std::endl;
      cpu_gpu_match = false;
      break;
    }
  }

  cudaStreamDestroy(kernelStream);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
  free(C_res_CPU);

  return cpu_gpu_match ? 0 : 1;
}
