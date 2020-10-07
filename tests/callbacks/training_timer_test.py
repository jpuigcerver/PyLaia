import pytest
import pytorch_lightning as pl
import torch

import laia.common.logging as log
from laia.callbacks import TrainingTimer
from laia.dummies import DummyModule, DummyTrainer


# classes outside of test because they need to be pickle-able
class __TestCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, *args):
        assert sum(isinstance(c, TrainingTimer) for c in trainer.callbacks) == 1
        for c in trainer.callbacks:
            if isinstance(c, TrainingTimer):
                assert isinstance(c.tr_timer.start, float)
                assert c.tr_timer.end is None

    def on_validation_epoch_start(self, trainer, *args):
        assert sum(isinstance(c, TrainingTimer) for c in trainer.callbacks) == 1
        for c in trainer.callbacks:
            if isinstance(c, TrainingTimer):
                assert isinstance(c.va_timer.start, float)
                assert c.va_timer.end is None


def _setup_logging(log_filepath):
    log.config(fmt="%(message)s", filename=log_filepath, filemode="w")


class __TestModule(DummyModule):
    def __init__(self, log_filepath, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_filepath = log_filepath
        _setup_logging(log_filepath)

    def configure_ddp(self, *args, **kwargs):
        # call _setup_logging again here otherwise processes
        # spawned by multiprocessing are not correctly configured
        _setup_logging(self.log_filepath)
        return super().configure_ddp(*args, **kwargs)


@pytest.fixture(scope="function", autouse=True)
def random_port():
    # these tests should run in separate processes,
    # otherwise addresses clash, fix by using a random port
    import os
    import random

    port = random.randint(12000, 19000)
    os.environ["MASTER_PORT"] = str(port)


@pytest.mark.parametrize("num_processes", (1, 2))
def test_cpu(tmpdir, num_processes):
    log_filepath = tmpdir / "log"
    module = __TestModule(log_filepath, batch_size=1)
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        callbacks=[TrainingTimer(), __TestCallback()],
        distributed_backend="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
    )
    trainer.fit(module)

    # caplog does not seem to work with multiprocessing.spawn
    # test logging on saved log file
    assert log_filepath.exists()
    if num_processes > 1:
        assert tmpdir.join("log.rank1").exists()
    lines = [l.strip() for l in log_filepath.readlines()]
    assert (
        sum(
            l.startswith(f"Epoch {e}: tr_time=")
            for l in lines
            for e in range(trainer.max_epochs)
        )
        == trainer.max_epochs
    )


@pytest.mark.skip("TODO: .fit() hangs")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multi-GPU test")
def test_gpu(tmpdir):
    log_filepath = tmpdir / "log"
    _setup_logging(log_filepath)
    module = DummyModule(batch_size=1)
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        callbacks=[TrainingTimer(), __TestCallback()],
        gpus=2,
        distributed_backend="ddp",
    )
    trainer.fit(module)

    assert log_filepath.exists()
    assert not tmpdir.join("log.rank1").exists()  # TODO: this should exist
    lines = [l.strip() for l in log_filepath.readlines()]
    assert (
        sum(
            l.startswith(f"Epoch {e}: tr_time=")
            for l in lines
            for e in range(trainer.max_epochs)
        )
        == trainer.max_epochs
    )
