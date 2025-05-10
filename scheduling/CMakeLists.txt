cmake_minimum_required(VERSION 3.18)
project(SchedulerBenchmark LANGUAGES CXX CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Set default values for build-time constants
set(SCHEDULER_TYPE "JLFP" CACHE STRING "Scheduler type")
set(THREADS_PER_BLOCK 256 CACHE STRING "Threads per block")
set(BLOCK_COUNT 80 CACHE STRING "Number of blocks")
set(TPC_SPLIT_DENOM 2 CACHE STRING "Number of TPCs requested")
set(TPC_SUBSET 1 CACHE STRING "Subset of TPCs requested")

# Set this to the path where libsmctrl.a and libsmctrl.h are located
set(LIBSMCTRL_DIR "/home/user/workspace/libraries/libsmctrl" CACHE PATH "Path to libsmctrl")

# Include the header path
include_directories(${LIBSMCTRL_DIR})

# Add the static library manually
add_library(smctrl STATIC IMPORTED)
set_target_properties(smctrl PROPERTIES
    IMPORTED_LOCATION "${LIBSMCTRL_DIR}/libsmctrl.a"
)

# Add compile definitions
add_compile_definitions(
    SCHEDULER_TYPE="${SCHEDULER_TYPE}"
    THREADS_PER_BLOCK=${THREADS_PER_BLOCK}
    BLOCK_COUNT=${BLOCK_COUNT}
    TPC_SPLIT_DENOM=${TPC_SPLIT_DENOM}
    TPC_SUBSET=${TPC_SUBSET}
)

# Gather all source files
set(SOURCES
    tasks/task.cu
    schedulers/schedulerBase/scheduler.cu
    schedulers/JLFPScheduler/JLFP.cu
    schedulers/dumbScheduler/dumbScheduler.cu
    schedulers/FCFSScheduler/FCFSScheduler.cu
    schedulers/asyncCompletionQueue/completionQueue.cu
    jobs/kernels/busyKernel.cu
    jobs/kernels/printKernel.cu
    jobs/kernels/vectorAdd.cu
    jobs/kernels/matrixMultiplication.cu
    jobs/jobBase/job.cu
    jobs/printJob/printJob.cu
    jobs/busyJob/busyJob.cu
    jobs/vectorAddJob/vectorAddJob.cu
    jobs/matrixMultiplicationJob/matrixMultiplicationJob.cu
    jobs/jobLaunchInformation/busyJobLaunchInformation.cu
    jobs/jobLaunchInformation/printJobLaunchInformation.cu
    jobs/jobLaunchInformation/vectorAddJobLaunchInformation.cu
    jobs/jobLaunchInformation/matrixMultiplicationJobLaunchInformation.cu
    common/helpFunctions.cu
    common/deviceProps.cu
    common/maskElement.cu
    executive/scheduling.cu
    executive/runbenchmark.cu
)

# add_executable(
#     benchmark_${SCHEDULER_TYPE}
#     ${SOURCES} schedulerTestBenchmark.cu
# )
# target_link_libraries(benchmark_${SCHEDULER_TYPE}
#     PRIVATE
#         smctrl
#         cuda
#         cudart
# )

# add_executable(
#     evaluation
#     ${SOURCES} evaluation.cu
# )
# target_link_libraries(evaluation
#     PRIVATE
#         smctrl
#         cuda
#         cudart
# )


add_executable(
    WCET_evaluation
    ${SOURCES} WCET_evaluation.cu
)
target_link_libraries(WCET_evaluation
    PRIVATE
        smctrl
        cuda
        cudart
)

add_executable(logsm logsm.cu)
target_link_libraries(logsm
    PRIVATE
        smctrl
        cuda
        cudart
)

add_executable(logsm2 logsm2.cu)
target_link_libraries(logsm2
        PRIVATE
        smctrl
        cuda
        cudart
)

add_executable(tpcscaling tpcscaling.cu)
target_link_libraries(tpcscaling
        PRIVATE
        smctrl
        cuda
        cudart
)

add_executable(tpc_parallel_benchmark tpc_parallel_benchmark.cu)
target_link_libraries(tpc_parallel_benchmark
    PRIVATE
        smctrl
        cuda
        cudart
)

# Link directories (libsmctrl)
link_directories(${LIBSMCTRL_DIR})
