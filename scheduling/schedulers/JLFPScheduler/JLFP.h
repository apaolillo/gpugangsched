#ifndef JLFP_H
#define JLFP_H
// JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique
// priority at launch time. Priorities can overlap, it is left to the scheduler
// on how to deal with this.
#include "../schedulerBase/scheduler.h"
#include <cinttypes>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <sys/types.h>

class JLFP : public BaseScheduler, public JobObserver {
private:
  struct jobQueue {
    int priorityLevel;
    std::queue<Job *> jobs;
    jobQueue(int level, std::queue<Job *> jobs)
        : priorityLevel(level), jobs(jobs) {}
  };

  std::vector<jobQueue> priorityQueue;

  jobQueue createNewJobQueue(Job *job);

public:
  void onJobCompletion(Job *job, float jobCompletionTime) override;
  void dispatch() override;
  void addJob(Job *job) override;
  void displayQueuePriorities();
  void displayQueueJobs();
};

#endif
