import os
import shutil
import subprocess
from collections import defaultdict
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import laia.common.logging as log

_logger = log.get_logger(__name__)


class GPUStats(pl.callbacks.GPUStatsMonitor):
    def _get_gpu_stats(self, gpu_stat_keys):
        gpu_query = ",".join([m[0] for m in gpu_stat_keys])
        format = "csv,nounits,noheader"

        result = subprocess.run(
            [
                shutil.which("nvidia-smi"),
                f"--query-gpu={gpu_query}",
                f"--format={format}",
                f"--id={self._gpu_ids}",
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True,
        )

        def _to_float(x):
            try:
                return float(x)
            except ValueError:
                return 0.0

        stats = result.stdout.strip().split(os.linesep)
        stats = [list(map(_to_float, x.split(", "))) for x in stats]

        # TODO: PR upstream to split this function
        logs = defaultdict(dict)
        for i, gpu_id in enumerate(self._gpu_ids.split(",")):
            for j, (key, unit) in enumerate(gpu_stat_keys):
                logs[gpu_id][key] = stats[i][j], unit
        return logs

    def on_train_batch_start(self, *args, **kwargs):
        pass

    @rank_zero_only
    def on_train_batch_end(self, trainer, *args, **kwargs):
        gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        gpu_stats = self._get_gpu_stats(gpu_stat_keys)
        log.debug("GPU stats: {}", gpu_stats)
        trainer.progress_bar_metrics["gpu_stats"] = GPUStats.parse_gpu_stats(gpu_stats)

    @staticmethod
    def parse_gpu_stats(logs: Dict) -> Dict[str, str]:
        out = {}
        for gpu_id in logs:
            mem_used, used_unit = logs[gpu_id]["memory.used"]
            mem_free, free_unit = logs[gpu_id]["memory.free"]
            mem_time, mem_time_unit = logs[gpu_id]["utilization.memory"]
            gpu_time, gpu_time_unit = logs[gpu_id]["utilization.gpu"]
            assert used_unit == free_unit
            out[f"GPU-{gpu_id}"] = (
                f"{int(mem_used)}/{int(mem_used + mem_free)}{used_unit}, "
                f"memory_time={int(mem_time)}{mem_time_unit}, "
                f"GPU_time={int(gpu_time)}{gpu_time_unit}"
            )
        return out
