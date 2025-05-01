from benchkit.benchmark import Benchmark
from pathlib import Path
from benchkit.platforms import Platform
from typing import List, Dict, Any

from benchkit.utils.dir import gitmainrootdir
from gpugangbenchutil.docker import GUEST_SRC_DIR
from gpugangbenchutil.taskgen import TaskSet, format_taskset


class SchedulerBenchmark(Benchmark):
    """Benchmark object for CUDA Scheduler comparisons."""

    def __init__(
        self,
        src_dir: Path,
        libsmctrl_dir: str,
        benchmark_file: str,
        scheduler_type: str,
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
        self._build_dir = self._bench_src_path / "build-benchkit"
        self._libsmctrl_dir = libsmctrl_dir
        self._benchmark_file = benchmark_file
        self._scheduler_type = scheduler_type
        self._results_path = self._bench_src_path / "results/scheduler_comparison"

    @property
    def bench_src_path(self) -> Path:
        return self._bench_src_path

    @staticmethod
    def get_build_var_names() -> List[str]:
        return ["threads_per_block", "block_count", "tpc_denom"]

    @staticmethod
    def get_run_var_names() -> List[str]:
        return []

    @staticmethod
    def get_tilt_var_names() -> List[str]:
        return []

    def dependencies(self) -> List:
        return super().dependencies() + []

    def build_bench(
        self,
        threads_per_block: int,
        block_count: int,
        tpc_denom: int,
        **kwargs,
    ) -> None:
        build_dir = self._build_dir
        self.platform.comm.makedirs(path=build_dir, exist_ok=True)

        build_command = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DSCHEDULER_TYPE={self._scheduler_type}",
            f"-DTHREADS_PER_BLOCK={threads_per_block}",
            f"-DTPC_SPLIT_DENOM={tpc_denom}",
            f"-DLIBSMCTRL_DIR={self._libsmctrl_dir}",
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

    def single_run(self, **kwargs) -> str:
        environment = self._preload_env(**kwargs)

        run_command = [f"./benchmark_{self._scheduler_type}"]

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
            print_output=True,
        )

        return output

    def parse_output_to_results(self, command_output: str, **kwargs) -> Dict[str, Any]:
        """Parse the output from the benchmark run into structured results."""
        results = {}

        # Extract metrics from the output
        lines = command_output.strip().split("\n")
        for line in lines:
            if "Execution time:" in line:
                time_str = line.split(":")[-1].strip()
                results["execution_time"] = float(time_str.split()[0])
            if "Average latency:" in line:
                latency_str = line.split(":")[-1].strip()
                results["average_latency"] = float(latency_str.split()[0])
            if "Throughput:" in line:
                throughput_str = line.split(":")[-1].strip()
                results["throughput"] = float(throughput_str.split()[0])
            if "Jobs missed deadline:" in line:
                deadline_miss_str = line.split(":")[-1].strip()
                results["deadline_misses"] = int(deadline_miss_str.split()[0])
            if "Jobs completed:" in line:
                jobs_completed_str = line.split(":")[-1].strip()
                results["jobs_completed"] = int(jobs_completed_str.split()[0])
            if "Total task system utilization:" in line:
                utilization_str = line.split(":")[-1].strip()
                results["task_system_utilization"] = float(utilization_str.split()[0])

        return results


class GangEvalBenchmark(SchedulerBenchmark):
    def __init__(
        self,
        src_dir: Path,
        libsmctrl_dir: str,
        platform: Platform,
        tasksets: List[TaskSet],
    ) -> None:
        super().__init__(
            src_dir=src_dir,
            libsmctrl_dir=libsmctrl_dir,
            benchmark_file="evaluation.cu",  # not used, placeholders
            scheduler_type="dumbScheduler",  # not used, placeholders
            platform=platform
        )
        self._tasksets = tasksets
        self._gmrd = gitmainrootdir()

    @staticmethod
    def get_build_var_names() -> List[str]:
        return ["tpc_denom"]

    @staticmethod
    def get_run_var_names() -> List[str]:
        return ["taskset_id", "scheduler"]

    def build_bench(
        self,
        tpc_denom: int,
        **kwargs,
    ) -> None:
        build_dir = self._build_dir
        self.platform.comm.makedirs(path=build_dir, exist_ok=True)

        build_command = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTPC_SPLIT_DENOM={tpc_denom}",
            f"-DLIBSMCTRL_DIR={self._libsmctrl_dir}",
            f"{self.bench_src_path}",
        ]

        # Compile the benchmark with the specific scheduler
        self.platform.comm.shell(
            command=build_command,
            current_dir=build_dir,
            output_is_log=True,
        )

        self.platform.comm.shell(
            command=["make", "-j", f"{self.platform.nb_active_cpus()}", "evaluation"],
            current_dir=build_dir,
            output_is_log=True,
        )

    def single_run(
        self,
        taskset_id: int,
        scheduler: str,
        record_data_dir: Path,
        **kwargs,
    ) -> str:
        taskset = self._tasksets[taskset_id]
        taskset_str = format_taskset(taskset=taskset)
        taskset_host_path = record_data_dir / f"taskset_{taskset_id}.txt"
        taskset_host_path.write_text(data=taskset_str + "\n")
        taskset_platform_path = GUEST_SRC_DIR / taskset_host_path.relative_to(self._gmrd)

        environment = self._preload_env(**kwargs)

        run_command = [
            f"./evaluation",
            "--scheduler",
            f"{scheduler}",
            "--task-file",
            f"{taskset_platform_path}",
        ]

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
            ignore_ret_codes=(1, 139),
            print_output=True,
        )

        return output
