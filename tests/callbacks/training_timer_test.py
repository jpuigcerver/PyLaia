import pytest
import pytorch_lightning as pl

from laia.callbacks import TrainingTimer
from laia.dummies import DummyEngine, DummyLoggingPlugin, DummyMNIST, DummyTrainer


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


@pytest.mark.parametrize("num_processes", (1, 2))
def test_cpu(tmpdir, num_processes):
    log_filepath = tmpdir / "log"
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        callbacks=[TrainingTimer(), __TestCallback()],
        accelerator="ddp_cpu" if num_processes > 1 else None,
        num_processes=num_processes,
        plugins=[DummyLoggingPlugin(log_filepath)],
    )
    trainer.fit(DummyEngine(), datamodule=DummyMNIST(batch_size=1))

    # caplog does not seem to work with multiprocessing.spawn
    # test logging on saved log file
    if num_processes > 1:
        log_filepath_rank1 = tmpdir.join("log.rank1")
        assert log_filepath_rank1.exists()
        assert not log_filepath_rank1.read_text("utf-8")

    assert log_filepath.exists()
    lines = [l.strip() for l in log_filepath.readlines()]
    assert (
        sum(
            l.startswith(f"E{e}: tr_time=")
            for l in lines
            for e in range(trainer.max_epochs)
        )
        == trainer.max_epochs
    )
