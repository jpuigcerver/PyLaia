import pytest
import pytorch_lightning as pl

import laia.common.logging as log
from laia.callbacks import TrainingTimer
from laia.dummies import DummyMNIST, DummyModule, DummyTrainer


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
    def __init__(self, log_filepath):
        super().__init__()
        self.log_filepath = log_filepath
        _setup_logging(log_filepath)

    def configure_ddp(self, *args, **kwargs):
        # call _setup_logging again here otherwise processes
        # spawned by multiprocessing are not correctly configured
        _setup_logging(self.log_filepath)
        return super().configure_ddp(*args, **kwargs)


@pytest.mark.parametrize("num_processes", (1, 2))
def test_cpu(tmpdir, num_processes):
    log_filepath = tmpdir / "log"
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        callbacks=[TrainingTimer(), __TestCallback()],
        distributed_backend="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
    )
    module = __TestModule(log_filepath)
    trainer.fit(module, datamodule=DummyMNIST(batch_size=1))

    # caplog does not seem to work with multiprocessing.spawn
    # test logging on saved log file
    if num_processes > 1:
        log_filepath_rank1 = tmpdir.join("log.rank1")
        assert log_filepath_rank1.exists()
        assert not len(log_filepath_rank1.read_text("utf-8"))

    assert log_filepath.exists()
    lines = [l.strip() for l in log_filepath.readlines()]
    assert (
        sum(
            l.startswith(f"Epoch {e}: tr_time=")
            for l in lines
            for e in range(trainer.max_epochs)
        )
        == trainer.max_epochs
    )
