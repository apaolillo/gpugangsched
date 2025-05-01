from benchkit.platforms import Platform
from gpus import (
    get_gpu_runner,
    get_gpu_docker_platform_from,
    get_gpu_builder,
)
from smctrl import install_libsmctrl_from_src


GUEST_SRC_DIR = "/home/user/src"


def get_docker_platform(
    host_src_dir: str,
) -> Platform:
    builder = get_gpu_builder()
    install_libsmctrl_from_src(
        builder=builder,
        workdir="/home/${USER_NAME}/workspace/libraries",
    )
    builder.build()
    runner = get_gpu_runner()

    docker_platform = get_gpu_docker_platform_from(
        runner=runner,
        host_src_dir=host_src_dir,
        guest_src_dir=GUEST_SRC_DIR,
    )

    return docker_platform
