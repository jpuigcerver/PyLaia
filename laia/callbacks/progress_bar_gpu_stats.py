from typing import Dict, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.utilities import DeviceType, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import laia.common.logging as log

_logger = log.get_logger(__name__)


class ProgressBarGPUStats(pl.callbacks.GPUStatsMonitor):
    def __init__(self):
        super().__init__(
            memory_utilization=True,
            gpu_utilization=False,
            intra_step_time=False,
            inter_step_time=False,
            fan_speed=False,
            temperature=False,
        )

    def on_train_start(self, trainer, *args, **kwargs):
        if not trainer._device_type == DeviceType.GPU:
            raise MisconfigurationException(
                "You are using GPUStatsMonitor but are not running on GPU"
                f" since gpus attribute in Trainer is set to {trainer.gpus}."
            )
        self._gpu_ids = ",".join(map(str, trainer.data_parallel_device_ids))

    def on_train_batch_start(self, *_, **__):
        pass

    @rank_zero_only
    def on_train_batch_end(self, trainer, *_, **__):
        gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        log.debug("GPU stats: {}", gpu_stats)
        progress_bar_metrics = ProgressBarGPUStats.parse_gpu_stats(
            self._gpu_ids, gpu_stats, gpu_stat_keys
        )
        trainer.progress_bar_metrics["gpu_stats"] = progress_bar_metrics

    @staticmethod
    def parse_gpu_stats(
        gpu_ids: str, stats: List[List[float]], keys: List[Tuple[str, str]]
    ) -> Dict[str, str]:
        j1, j2, used_unit, free_unit = None, None, None, None
        for i, k in enumerate(keys):
            if k[0] == "memory.used":
                j1, used_unit = i, k[1]
            elif k[0] == "memory.free":
                j2, free_unit = i, k[1]

        assert j1 is not None and j2 is not None
        assert used_unit == free_unit
        gpu_ids = gpu_ids.split(",")
        assert len(gpu_ids) == len(stats)

        return {
            f"GPU-{gpu_id}": f"{int(stats[i][j1])}/{int(stats[i][j1] + stats[i][j2])}{used_unit}"
            for i, gpu_id in enumerate(gpu_ids)
        }
