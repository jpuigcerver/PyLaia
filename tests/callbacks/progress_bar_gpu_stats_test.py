import shutil

from laia.callbacks import ProgressBarGPUStats
from laia.dummies import DummyEngine, DummyMNIST, DummyTrainer


def test_parse_gpu_stats():
    gpu_ids = "0,1"
    gpu_stats = [[0.0, 0.0, 0.0, 0.0], [3287.3, 4695.0, 8.0, 16]]
    gpu_stat_keys = [
        ("memory.used", "MB"),
        ("memory.free", "MB"),
        ("utilization.memory", "%"),
        ("utilization.gpu", "%"),
    ]
    expected = {"GPU-0": "0/0MB", "GPU-1": "3287/7982MB"}
    assert (
        ProgressBarGPUStats.parse_gpu_stats(gpu_ids, gpu_stats, gpu_stat_keys)
        == expected
    )


def test_progress_bar_gpu_stats(monkeypatch, tmpdir):
    def _fake_on_train_start(self, *_):
        self._gpu_ids = "0,1"

    fake_stats = [[1.2, 2.3], [3.4, 4.5]]
    monkeypatch.setattr(shutil, "which", lambda _: True)
    monkeypatch.setattr(ProgressBarGPUStats, "on_train_start", _fake_on_train_start)
    monkeypatch.setattr(ProgressBarGPUStats, "_get_gpu_stats", lambda *_: fake_stats)

    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[ProgressBarGPUStats()],
        progress_bar_refresh_rate=1,
    )
    trainer.fit(DummyEngine(), datamodule=DummyMNIST())

    expected = {
        f"GPU-{i}": f"{int(fake_stats[i][0])}/{int(sum(fake_stats[i]))}MB"
        for i in range(2)
    }
    assert trainer.progress_bar_dict["gpu_stats"] == expected
