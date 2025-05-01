#!/bin/bash

LIBSMCTRL_PATH="../commonLib/libsmctrl/"

#ls -l $LIBSMCTRL_PATH

nvcc -g -G -o main main.cu tasks/task.cu schedulers/schedulerBase/scheduler.cu schedulers/JLFPScheduler/JLFP.cu schedulers/dumbScheduler/dumbScheduler.cu \
  schedulers/asyncCompletionQueue/completionQueue.cu schedulers/FCFSScheduler/FCFSScheduler.cu jobs/kernels/busyKernel.cu jobs/kernels/vectorAdd.cu \
  jobs/kernels/matrixMultiplication.cu \
  jobs/jobLaunchInformation/printJobLaunchInformation.cu jobs/jobLaunchInformation/busyJobLaunchInformation.cu jobs/jobLaunchInformation/vectorAddJobLaunchInformation.cu \
  jobs/jobLaunchInformation/matrixMultiplicationJobLaunchInformation.cu \
  jobs/kernels/printKernel.cu jobs/jobBase/job.cu jobs/printJob/printJob.cu jobs/busyJob/busyJob.cu jobs/vectorAddJob/vectorAddJob.cu jobs/matrixMultiplicationJob/matrixMultiplicationJob.cu \
  common/helpFunctions.cu common/deviceProps.cu common/maskElement.cu \

  -I${LIBSMCTRL_PATH} -lsmctrl -lcuda -L${LIBSMCTRL_PATH}
  