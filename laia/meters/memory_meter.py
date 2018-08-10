import resource

import torch
import torch.cuda as cuda

from laia.meters import Meter


class MemoryMeter(Meter):
    def __init__(self, device=None, exceptions_threshold=5):
        # type: (torch.device, int) -> None
        super(MemoryMeter, self).__init__(exceptions_threshold)
        self._device = device

    def get_cuda_memory(self):
        # type: () -> str
        # Convert from B to MiB
        if cuda.is_available():
            cuda.empty_cache()
            return "{:.0f} MiB".format(
                cuda.max_memory_cached(device=self._device) // 1024 ** 2
            )
        return "??? MiB"

    def get_cpu_memory(self):
        # type: () -> str
        # Convert from KB to MiB
        return "{:.0f} MiB".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10 ** 3 / 2 ** 20
        )

    @property
    def value(self):
        # type: () -> str
        if self._device is None:
            name = "get_cuda_memory" if cuda.is_available() else "get_cpu_memory"
        else:
            name = "get_{}_memory".format(self._device.type)
        return getattr(self, name)()
