from benchkit.campaign import CampaignCartesianProduct, CampaignSuite
from benchkit.utils.dir import gitmainrootdir
from gpugangbenchutil.docker import get_docker_platform, GUEST_SRC_DIR
from gpugangbenchutil.benchmark import GangEvalBenchmark
from pathlib import Path
import random
from gpugangbenchutil.taskgen import generate_taskset


seed = 42
nb_ts_per_u = 5
benchmark_duration_seconds=3
nb_tasks = 5
utilizations = [0.5, 0.75, 0.9]
schedulers = [
    "JLFP",
    "FCFS",
    "dumbScheduler",
]
job_types = [
    "print",
    "busy",
]

def main() -> None:
    gmrd = gitmainrootdir()
    platform = get_docker_platform(host_src_dir=str(gmrd))
    libsmctrl_dir = "/home/user/workspace/libraries/libsmctrl"
    src_dir = Path(GUEST_SRC_DIR)

    rng = random.Random(seed)  # Seeded RNG for reproducibility

    tasksets = [generate_taskset(
        num_tasks=nb_tasks,
        total_util=u,
        job_types=job_types,
        rng=rng,
        min_period=20,
        max_period=100,
        threads_per_block=128,
        block_count=4,
    ) for u in utilizations for _ in range(nb_ts_per_u)]

    bench = GangEvalBenchmark(
        src_dir=src_dir / "scheduling",
        libsmctrl_dir=libsmctrl_dir,
        platform=platform,
        tasksets=tasksets,
    )

    campaign = CampaignCartesianProduct(
        name="gang",
        benchmark=bench,
        nb_runs=1,
        variables={
            "tpc_denom": [2],#[2, 4]
            "taskset_id": range(len(tasksets)),
            "scheduler": schedulers,
        },
        constants={"seed": seed},
        debug=False,
        gdb=False,
        enable_data_dir=True,
        benchmark_duration_seconds=benchmark_duration_seconds,
    )

    suite = CampaignSuite(campaigns=[campaign])
    suite.print_durations()
    suite.run_suite()


if __name__ == "__main__":
    main()
