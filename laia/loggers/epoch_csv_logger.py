import csv
import os
import re
from collections import defaultdict
from typing import Optional, Union

from pytorch_lightning.loggers.csv_logs import CSVLogger, ExperimentWriter
from pytorch_lightning.utilities import rank_zero_only


class EpochCSVWriter(ExperimentWriter):
    def save(self, version: Optional[int] = None) -> None:
        if not self.metrics:
            return

        metrics = self.group_by_epoch(self.metrics)
        keys = sorted({k for m in metrics for k in m.keys()})
        file_path = os.path.join(
            self.log_dir, "metrics.csv" if version < 0 else f"metrics-v{version}.csv"
        )

        with open(file_path, "w", newline="") as f:
            self.writer = csv.DictWriter(f, fieldnames=keys)
            self.writer.writeheader()
            self.writer.writerows(metrics)

    @staticmethod
    def group_by_epoch(metrics):
        # filter out 'step'
        filtered = []
        for m in metrics:
            if "epoch" in m:
                m.pop("step", None)
            else:
                step = m.pop("step", None)
                if step is not None:
                    m["epoch"] = step
            filtered.append(m)

        # merge dicts by epoch
        out = defaultdict(dict)
        for d in filtered:
            if "epoch" in d:
                out[d["epoch"]].update(d)
        return [v for _, v in sorted(out.items())]


class EpochCSVLogger(CSVLogger):
    def __init__(self, save_dir: str, version: Optional[Union[int, str]] = None):
        super().__init__(save_dir, name=None, version=version)
        self._experiment = None

    @property
    def log_dir(self) -> str:
        return self.root_dir

    @property
    def experiment(self) -> EpochCSVWriter:
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = EpochCSVWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def save(self) -> None:
        super(CSVLogger, self).save()
        self.experiment.save(version=self.version)

    def _get_next_version(self) -> Optional[int]:
        return self.get_next_version(self.root_dir)

    @staticmethod
    def get_next_version(root_dir):
        versions = []
        for d in os.listdir(root_dir):
            if d == "metrics.csv":
                # first csv
                versions.append(0)
                continue
            match = re.match(r"metrics-v(\d+)\.csv", d)
            if match:
                versions.append(int(match.group(1)) + 1)

        if len(versions) == 0:
            # no csv file yet
            return -1

        return max(versions)
