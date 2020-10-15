import re

import pytorch_lightning as pl

from laia.callbacks import ProgressBar
from laia.dummies import DummyEngine, DummyMNIST, DummyTrainer


class __TestCallback(pl.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar

    def on_epoch_start(self, trainer, *_, **__):
        assert self.pbar.main_progress_bar.total == trainer.num_training_batches
        assert self.pbar.main_progress_bar.desc.startswith("TR - Epoch ")

    def on_train_epoch_start(self, *_, **__):
        assert self.pbar.tr_timer.end is None

    def on_validation_epoch_start(self, trainer, *_, **__):
        if trainer.running_sanity_check:
            assert self.pbar.val_progress_bar.desc.startswith("VA sanity check")
        else:
            assert self.pbar.tr_timer.end is not None
            assert self.pbar.va_timer.end is None
            assert self.pbar.val_progress_bar.desc.startswith("VA - Epoch")

    def on_train_batch_end(self, *_, **__):
        assert "running_loss" in str(self.pbar.main_progress_bar)
        assert "gpu_stats" in str(self.pbar.main_progress_bar)

    def on_train_epoch_end(self, *_, **__):
        assert "cer=100.0%" in str(self.pbar.main_progress_bar)
        assert "wer=33.0" in str(self.pbar.val_progress_bar)

    def on_validation_epoch_end(self, trainer, *_, **__):
        if not trainer.running_sanity_check:
            assert self.pbar.va_timer.end is not None


def test_progress_bar(tmpdir):
    pbar = ProgressBar()
    module = DummyEngine()
    data_module = DummyMNIST()
    trainer = DummyTrainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        callbacks=[pbar, __TestCallback(pbar)],
    )

    # fake things to include in the pbar
    trainer.progress_bar_metrics["tr_cer"] = 1
    trainer.progress_bar_metrics["va_wer"] = 0.33
    trainer.progress_bar_metrics["gpu_stats"] = {"gpu_stats": "baz"}

    trainer.fit(module, datamodule=data_module)
    n, m = trainer.limit_train_batches, trainer.limit_val_batches
    assert pbar.is_enabled
    # check counts
    assert pbar.total_train_batches == pbar.main_progress_bar.total == n
    assert pbar.total_val_batches == pbar.val_progress_bar.total == m
    # check end was reached
    assert pbar.main_progress_bar.n == pbar.train_batch_idx == n
    assert pbar.val_progress_bar.n == pbar.val_batch_idx == m
    # check test bar is off
    assert pbar.total_test_batches == 0
    assert pbar.test_progress_bar is None
    # check bar string
    float_pattern = "([0-9]*[.])?[0-9]+"
    assert re.match(
        rf"TR - Epoch 1: 100%\|[█]+\| 10\/10 \[00:00<00:00, {float_pattern}it\/s\, cer=100\.0%]",
        str(pbar.main_progress_bar),
    )
    assert re.match(
        rf"VA - Epoch 1: 100%\|[█]+\| 10\/10 \[00:00<00:00, {float_pattern}it\/s\, wer=33\.0%]",
        str(pbar.val_progress_bar),
    )

    trainer.test(module, datamodule=data_module)
    # previous checks for test
    k = trainer.limit_test_batches
    assert pbar.total_test_batches == pbar.test_progress_bar.total == k
    assert pbar.test_progress_bar.n == pbar.test_batch_idx == k
    assert re.match(
        rf"Decoding: 100%\|[█]+\| 10\/10 \[00:00<00:00, {float_pattern}it\/s\]",
        str(pbar.test_progress_bar),
    )
