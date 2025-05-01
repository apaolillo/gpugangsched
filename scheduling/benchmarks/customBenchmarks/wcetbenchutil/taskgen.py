import random
from typing import List, Literal, Optional, Union, Tuple

JobType = str

Task = Tuple[
    JobType,
    int,
    int,
    int,
    int,
    int,
    int,
]

TaskSet = List[Task]


def uunifast(
    n: int,
    u_total: float,
    rng: random.Random,
) -> List[float]:
    """Generate n utilizations summing to U_total using the UUniFast algorithm."""
    utilizations: List[float] = []
    sum_u = u_total
    for i in range(1, n):
        next_sum_u = sum_u * (rng.random() ** (1.0 / (n - i)))
        utilizations.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    utilizations.append(sum_u)
    return utilizations


def generate_taskset(
    num_tasks: int,
    total_util: float,
    job_types: List[JobType],
    rng: Optional[random.Random] = None,
    min_period: int = 20,
    max_period: int = 100,
    threads_per_block: int = 128,
    block_count: int = 4
) -> TaskSet:
    """Generate a taskset for the GPU scheduler.

    Args:
        num_tasks: Number of tasks to generate.
        total_util: Target total utilization (e.g., 0.75).
        job_types: List of task types to randomly choose from.
        rng: Optional seeded random number generator.
        min_period: Minimum period for a task.
        max_period: Maximum period for a task.
        threads_per_block: Threads per CUDA block.
        block_count: Number of blocks.

    Returns:
        A list of task descriptors, each a list of [type, offset, wcet, deadline, period, tpb, blocks].
    """
    if rng is None:
        rng = random.Random()

    utilizations = uunifast(num_tasks, total_util, rng)
    taskset: TaskSet = []

    for u in utilizations:
        job_type = rng.choice(job_types)
        period = rng.randint(min_period, max_period)
        wcet = max(1, int(u * period))
        deadline = rng.randint(wcet, period)
        offset = 0

        task: Task = (
            job_type,
            offset,
            wcet,
            deadline,
            period,
            threads_per_block,
            block_count,
        )
        taskset.append(task)

    return taskset


def format_taskset(taskset: TaskSet) -> str:
    """Flatten and format a taskset to match your scheduler's input format."""
    flat: List[str] = [str(len(taskset))]
    for task in taskset:
        flat.extend(map(str, task))
    return " ".join(flat)


def get_utilization(task: Task) -> float:
    return task[2] / task[4]


def get_total_utilization(taskset: TaskSet) -> float:
    return sum([get_utilization(task=t) for t in taskset])


def main():
    rng = random.Random(42)  # Seeded RNG for reproducibility
    for i in range(100):
        ts = generate_taskset(
            num_tasks=5,
            total_util=0.75,
            job_types=["print", "busy"],
            rng=rng
        )
        print(ts)
        util = sum([t[2] / t[4] for t in ts])
        print(util)
        print(get_total_utilization(ts))
        print()


if __name__ == "__main__":
    main()
