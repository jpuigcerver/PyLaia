import pytest
import torch

from laia.callbacks import LearningRate
from laia.dummies import DummyEngine, DummyLoggingPlugin, DummyMNIST, DummyTrainer


def test_learning_rate_warns(tmpdir):
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[LearningRate()],
    )
    with pytest.warns(RuntimeWarning, match=r"You are using LearningRateMonitor.*"):
        trainer.fit(DummyEngine(), datamodule=DummyMNIST())


class __TestEngine(DummyEngine):
    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, 1)]


# TODO: fix test with num_processes=2
@pytest.mark.parametrize("num_processes", (1,))
def test_learning_rate(tmpdir, num_processes):
    log_filepath = tmpdir / "log"

    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        callbacks=[LearningRate()],
        accelerator="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
        plugins=[DummyLoggingPlugin(log_filepath)],
    )
    trainer.fit(__TestEngine(), datamodule=DummyMNIST())

    if num_processes > 1:
        log_filepath_rank1 = tmpdir.join("log.rank1")
        assert log_filepath_rank1.exists()
        assert not log_filepath_rank1.read_text("utf-8")

    assert log_filepath.exists()
    lines = [l.strip() for l in log_filepath.readlines()]
    for e in range(1, trainer.max_epochs):
        expected = f"E{e}: lr-Adam 1.000e-0{e + 2} ⟶ 1.000e-0{e + 3}"
        # Additional UserWarning present till https://github.com/Lightning-AI/pytorch-lightning/pull/6139 (lightning 1.3.0)
        assert lines.count(expected) == 2
