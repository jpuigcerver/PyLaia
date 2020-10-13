from pytorch_lightning.accelerators.ddp_cpu_spawn_accelerator import (
    DDPCPUSpawnAccelerator,
)
from pytorch_lightning.cluster_environments import ClusterEnvironment

import laia.common.logging as log


def _setup_logging(log_filepath):
    log.config(fmt="%(message)s", filename=log_filepath, filemode="w")


def dummy_accelerator_args(log_filepath, nprocs):
    _setup_logging(log_filepath)
    return {
        "num_processes": nprocs,
        "accelerator": DummyLoggingAccelerator(log_filepath, nprocs)
        if nprocs > 1
        else None,
    }


class DummyEnvironment(ClusterEnvironment):
    def master_address(self):
        return "127.0.0.1"  # localhost

    def master_port(self):
        return "8080"


class DummyLoggingAccelerator(DDPCPUSpawnAccelerator):
    def __init__(self, log_filepath, nprocs=2):
        environment = DummyEnvironment()
        environment._world_size = nprocs
        super().__init__(trainer=None, nprocs=nprocs, cluster_environment=environment)
        self.log_filepath = log_filepath

    def configure_ddp(self, *args, **kwargs):
        # call _setup_logging again here otherwise processes
        # spawned by multiprocessing are not correctly configured
        _setup_logging(self.log_filepath)
        return super().configure_ddp(*args, **kwargs)
