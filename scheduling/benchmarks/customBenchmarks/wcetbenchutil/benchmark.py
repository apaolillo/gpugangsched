from benchkit.benchmark import Benchmark
from pathlib import Path
from benchkit.platforms import Platform
from typing import List, Dict, Any

from benchkit.utils.dir import gitmainrootdir
from wcetbenchutil.docker import GUEST_SRC_DIR
from wcetbenchutil.taskgen import TaskSet, format_taskset


class WCETEvalBenchmark(Benchmark):
    """Benchmark object for CUDA Scheduler comparisons."""

    def __init__(
        self,
        src_dir: Path,
        libsmctrl_dir: str,
        platform: Platform = None,
    ) -> None:
        super().__init__(
            command_wrappers=(),
            command_attachments=(),
            shared_libs=(),
            pre_run_hooks=(),
            post_run_hooks=(),
        )

        if platform is not None:
            self.platform = platform

        self._bench_src_path = Path(src_dir)
        self._build_dir = self._bench_src_path / "build"
        self._libsmctrl_dir = libsmctrl_dir
        self._results_path = self._bench_src_path / "results/wcet_comparaison"

    @property
    def bench_src_path(self) -> Path:
        return self._bench_src_path

    @staticmethod
    def get_build_var_names() -> List[str]:
        return []

    @staticmethod
    def get_run_var_names() -> List[str]:
        return ["vector_size", "tpc_denom", "threads_per_block"]

    @staticmethod
    def get_tilt_var_names() -> List[str]:
        return []

    def dependencies(self) -> List:
        return super().dependencies() + []

    def build_bench(
        self,
        **kwargs,
    ) -> None:
        build_dir = self._build_dir
        self.platform.comm.makedirs(path=build_dir, exist_ok=True)

        build_command = [
            "cmake",
            f"{self.bench_src_path}",
        ]

        # Compile the benchmark with the specific scheduler
        self.platform.comm.shell(
            command=build_command,
            current_dir=build_dir,
            output_is_log=True,
        )

        self.platform.comm.shell(
            command=["make", "-j", f"{self.platform.nb_active_cpus()}"],
            current_dir=build_dir,
            output_is_log=True,
        )

    def clean_bench(self) -> None:
        pass

    def single_run(self,
        tpc_denom: int,
        vector_size: int,
        threads_per_block: int,
        **kwargs
    ) -> str:
        environment = self._preload_env(**kwargs)

        run_command = [f"./WCET_evaluation", f"{vector_size}", f"{tpc_denom}", f"{threads_per_block}"]

        wrapped_run_command, wrapped_environment = self._wrap_command(
            run_command=run_command,
            environment=environment,
            **kwargs,
        )

        output = self.run_bench_command(
            run_command=run_command,
            wrapped_run_command=wrapped_run_command,
            current_dir=self._build_dir,
            environment=environment,
            wrapped_environment=wrapped_environment,
            ignore_ret_codes=[1],
            print_output=True,
        )

        return output

    def parse_output_to_results(self, command_output: str, **kwargs) -> Dict[str, Any]:
        """Parse the output from the benchmark run into structured results."""
        results = {}

        # Extract metrics from the output
        lines = command_output.strip().split("\n")
        for line in lines:
            if "CPU computation time:" in line:
                time_str = line.split(":")[-1].strip()
                results["cpu_execution_time"] = float(time_str.split()[0])
            if "Number of TPCs: " in line:
                latency_str = line.split(":")[-1].strip()
                results["number_tpc"] = float(latency_str.split()[0])
            if "GPU computation time (without. copy & sync):" in line:
                throughput_str = line.split(":")[-1].strip()
                results["gpu_execution_kernel"] = float(throughput_str.split()[0])
            if "GPU computation time (incl. copy & sync):" in line:
                deadline_miss_str = line.split(":")[-1].strip()
                results["gpu_execution_all"] = float(deadline_miss_str.split()[0])
            if "GPU synchronization time:" in line:
                jobs_completed_str = line.split(":")[-1].strip()
                results["gpu_execution_synchronization"] = float(jobs_completed_str.split()[0])

        return results
