import pytest
import torch

import laia.common.logging as log
from laia.callbacks import LearningRate
from laia.dummies import DummyMNIST, DummyModule, DummyTrainer


def test_learning_rate_warns(tmpdir):
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[LearningRate()],
    )
    with pytest.warns(RuntimeWarning, match=r"You are using LearningRateMonitor.*"):
        trainer.fit(DummyModule(), datamodule=DummyMNIST())


def _setup_logging(log_filepath):
    log.config(fmt="%(message)s", filename=log_filepath, filemode="w")


class __TestModule(DummyModule):
    def __init__(self, log_filepath):
        super().__init__()
        self.log_filepath = log_filepath
        _setup_logging(log_filepath)

    def configure_ddp(self, *args, **kwargs):
        _setup_logging(self.log_filepath)
        return super().configure_ddp(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, 1)]


@pytest.mark.parametrize("num_processes", (1, 2))
def test_learning_rate(tmpdir, num_processes):
    log_filepath = tmpdir / "log"
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        callbacks=[LearningRate()],
        num_processes=num_processes,
        distributed_backend="ddp_cpu" if num_processes > 1 else None,
    )
    module = __TestModule(log_filepath)
    trainer.fit(module, datamodule=DummyMNIST())

    if num_processes > 1:
        log_filepath_rank1 = tmpdir.join("log.rank1")
        assert log_filepath_rank1.exists()
        assert not len(log_filepath_rank1.read_text("utf-8"))

    assert log_filepath.exists()
    lines = [l.strip() for l in log_filepath.readlines()]
    for e in range(1, trainer.max_epochs):
        expected = f"Epoch {e}: lr-Adam 1.000e-0{e + 2} ⟶ 1.000e-0{e + 3}"
        assert lines.count(expected) == 1
