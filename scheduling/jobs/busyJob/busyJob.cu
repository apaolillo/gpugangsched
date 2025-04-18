#include "busyJob.h"
#include <cmath>
#include <cstdint>

// callback that is envoked at the end of each kernel execution.
void CUDART_CB BusyJob::busyKernelCallback(cudaStream_t stream,
                                           cudaError_t status, void *data) {

  // get the kernel launch config that has to be cleaned up and that contains
  // info to display.
  BusyJob::busyKernelLaunchInformation *kernelInfo =
      static_cast<BusyJob::busyKernelLaunchInformation *>(data);

  // free the dynamically allocated memory and the stream.
  cudaFreeHost(kernelInfo->hostPtr);
  cudaFree(kernelInfo->devicePtr);
  cudaFree(kernelInfo->timerDptr);
  cudaStreamDestroy(stream);
  // std::cout << "busy job finished\n";
  float currentTime = getCurrentTime();

  Job::notifyJobCompletion(kernelInfo->jobPtr, currentTime);
}

// callback constructor.
void BusyJob::addBusyKernelCallback(Job *job, cudaStream_t stream, float *dptr,
                                    float *hptr, size_t size, int id) {

  BusyJob::busyKernelLaunchInformation *kernelInfo =
      new BusyJob::busyKernelLaunchInformation(job, stream, dptr, hptr, size,
                                               id);

  cudaStreamAddCallback(stream, busyKernelCallback, kernelInfo, 0);
}

void BusyJob::execute() {

  // Allocate memory
  float *d_output;
  cudaMalloc(&d_output, sizeof(float));

  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);
  // set the stream's mask using libsmctrl.
  if (!this->TPCMasks.empty()) {
    std::cout << "mask set";
    uint64_t mask = this->combineMasks();
    libsmctrl_set_stream_mask(kernel_stream, mask);
  }
  // allocate host memory using cuda. If done this way, copying from device to
  // host can be done asynchronously.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  float *h_output = nullptr;
  size_t nbrOfBytes = sizeof(float);
  cudaHostAlloc((void **)&h_output, nbrOfBytes, cudaHostAllocDefault);

  maxUtilizationKernel<<<1, 1, 0, kernel_stream>>>(d_output, 1000);
  // define the asynchronous memory transfer here.
  cudaMemcpyAsync(h_output, d_output, nbrOfBytes, cudaMemcpyHostToDevice,
                  kernel_stream);
  // this callback is called only when the kernel is finished and the memory
  // copying is finished.
  addBusyKernelCallback(this, kernel_stream, d_output, h_output, sizeof(float),
                        1);
}

BusyJob::BusyJob(int threadsPerBlock, int threadBlocks) {
  this->threadsPerBlock = threadsPerBlock;
  this->threadBlocks = threadBlocks;

  int totalThreads = threadsPerBlock * threadBlocks;
  int neededSMs =
      totalThreads / DeviceInfo::getDeviceProps()->getMaxThreadsPerSM();

  // if the job needs les than one SM, assign it only one TPC. Included this if
  // statement because the devision did go to zero if the amount of threads is
  // too small.
  if (neededSMs < 1) {
    this->neededTPCs = 1;
    return;
  }
  this->neededTPCs =
      ceil(neededSMs / DeviceInfo::getDeviceProps()->getSMsPerTPC());
}
