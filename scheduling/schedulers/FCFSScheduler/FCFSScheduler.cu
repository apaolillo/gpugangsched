#include "FCFSScheduler.h"

void FCFSScheduler::dispatch() {
  while (!this->jobQueue.empty()) {
    Job *currJob = this->jobQueue.front();
    int totalTPCs = DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice();
    int neededTPCs = currJob->getNeededTPCs();
    if (neededTPCs > totalTPCs && neededTPCs < 2 * totalTPCs) {
      // in this case the job gets halve of the total TPCs.
      int allowedTPCs = totalTPCs / 2;
      setJobTPCMask(allowedTPCs, currJob);
      currJob->setJobObserver(this);
      currJob->execute();
      // std::cout << "launched with 1/2 of the total TPCs\n";
    } else if (neededTPCs >= 2 * totalTPCs) {
      setJobTPCMask(totalTPCs, currJob);
      currJob->setJobObserver(this);
      currJob->execute();
      //      std::cout << "launched with all TPCs\n";
    } else {
      setJobTPCMask(neededTPCs, currJob);
      currJob->setJobObserver(this);
      currJob->execute();
      //    std::cout << "launched a job with its needed TPCs\n";
    }
    this->jobQueue.pop();
  }
}

void FCFSScheduler::addJob(Job *job) { this->jobQueue.push(job); }

void FCFSScheduler::onJobCompletion(Job *job, float jobCompletionTime) {
  job->releaseMasks();
  // check if the job met its deadline.
  job->releaseMasks();
  if (!(job->getReleaseTime() + job->getAbsoluteDeadline() <
        jobCompletionTime)) {
    this->incDeadlineMisses();
  }
  this->incJobsCompleted();
}
