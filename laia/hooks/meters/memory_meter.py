from __future__ import absolute_import

import os
import resource
import subprocess
import sys

import torch

from laia.hooks.meters import Meter


class MemoryMeter(Meter):
    def __init__(self, exceptions_threshold=5):
        # type: (int) -> None
        super(MemoryMeter, self).__init__(exceptions_threshold)

    def get_gpu_memory(self):
        # type: () -> str

        result = str(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader",
                ]
            )
            .decode(sys.stdout.encoding)
            .strip()
        )
        for out in result.split("\n"):
            pid, mem = out.split(", ")
            if int(pid) == os.getpid():
                return mem

    def get_cpu_memory(self):
        # type: () -> str
        # Convert from KB to MiB
        return "{:.0f} MiB".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10 ** 3 / 2 ** 20
        )

    @property
    def value(self):
        # type: () -> str
        # TODO: GPU or CPU should be a parameter
        return (
            self.get_gpu_memory()
            if torch.cuda.is_available()
            else self.get_cpu_memory()
        )
