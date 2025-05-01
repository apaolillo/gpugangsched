from benchkit.campaign import CampaignCartesianProduct, CampaignSuite
from benchkit.utils.dir import gitmainrootdir
from wcetbenchutil.docker import get_docker_platform, GUEST_SRC_DIR
from wcetbenchutil.benchmark import WCETEvalBenchmark
from pathlib import Path
import random

vector_sizes = [i * 10_000_000 for i in range(10, 60, 10)] #[i * 1_000_000 for i in range(1, 501, 100)]
tpc_nb = range(1, 13)
threads_per_block = [128]
benchmark_duration_seconds=30

def main() -> None:
    gmrd = gitmainrootdir()
    platform = get_docker_platform(host_src_dir=str(gmrd))
    libsmctrl_dir = "/home/user/workspace/libraries/libsmctrl"
    src_dir = Path(GUEST_SRC_DIR)

    bench = WCETEvalBenchmark(
        src_dir=src_dir / "scheduling/",
        libsmctrl_dir=libsmctrl_dir,
        platform=platform,
    )

    campaign = CampaignCartesianProduct(
        name="gang",
        benchmark=bench,
        nb_runs=1,
        variables={
            "vector_size": vector_sizes,
            "tpc_denom": tpc_nb,#[2, 4]
            "threads_per_block": threads_per_block
        },
        constants=[],
        debug=False,
        gdb=False,
        enable_data_dir=True,
        benchmark_duration_seconds=benchmark_duration_seconds,
    )

    suite = CampaignSuite(campaigns=[campaign])
    suite.print_durations()
    suite.run_suite()
    
    campaign.generate_graph(
        title="Title",
        plot_name="lineplot",
        x="vector_size",
        y="gpu_execution_kernel",
        hue="tpc_denom",
        marker="o",
        markers=True,
        dashes=False,
    )

if __name__ == "__main__":
    main()
