import re

import pytorch_lightning as pl
from tqdm import tqdm

from laia.callbacks import ProgressBar
from laia.callbacks.meters import Timer
from laia.dummies import DummyEngine, DummyMNIST, DummyTrainer


class __TestCallback(pl.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar

    def on_epoch_start(self, trainer, *_, **__):
        assert self.pbar.main_progress_bar.total == trainer.num_training_batches
        assert self.pbar.main_progress_bar.desc.startswith("TR - E")

    def on_train_epoch_start(self, *_, **__):
        assert self.pbar.tr_timer.end is None

    def on_validation_epoch_start(self, trainer, *_, **__):
        if trainer.running_sanity_check:
            assert self.pbar.val_progress_bar.desc.startswith("VA sanity check")
        else:
            assert self.pbar.tr_timer.end is not None
            assert self.pbar.va_timer.end is None
            assert self.pbar.val_progress_bar.desc.startswith("VA - E")

    def on_train_batch_end(self, *_, **__):
        assert "running_loss" in str(self.pbar.main_progress_bar)
        assert "gpu_stats" in str(self.pbar.main_progress_bar)

    def on_train_epoch_end(self, *_, **__):
        assert "cer=100.0%" in str(self.pbar.main_progress_bar)
        assert "cer=33.0" in str(self.pbar.val_progress_bar)

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
    trainer.progress_bar_metrics["va_cer"] = 0.33
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
    pattern = (
        r" - E1: "
        r"100%\|[█]+\| 10/10 \[00:00<00:00, "
        rf"{float_pattern}it/s, "
        rf"loss={float_pattern}, "
        rf"cer={float_pattern}%, "
        r"gpu_stats={'gpu_stats': 'baz'}]"
    )
    assert re.match("TR" + pattern, str(pbar.main_progress_bar))
    assert re.match("VA" + pattern, str(pbar.val_progress_bar))

    trainer.test(module, datamodule=data_module)
    # previous checks for test
    k = trainer.limit_test_batches
    assert pbar.total_test_batches == pbar.test_progress_bar.total == k
    assert pbar.test_progress_bar.n == pbar.test_batch_idx == k
    assert re.match(
        rf"Decoding: 100%\|[█]+\| 10/10 \[00:00<00:00, {float_pattern}it/s]",
        str(pbar.test_progress_bar),
    )


def test_fix_format_dict():
    pbar = tqdm(desc="Test", postfix={"foo": "bar"}, total=30)
    timer = Timer()
    timer.end = timer.start + 50
    format_dict = ProgressBar.fix_format_dict(pbar, timer=timer)
    format_dict["rate"] = 1.55
    assert (
        tqdm.format_meter(**format_dict)
        == "Test: 0% 0/30 [00:50<00:19, 1.55it/s, foo=bar]"
    )
    format_dict["n"] = 29
    format_dict["rate"] = 10
    assert (
        tqdm.format_meter(**format_dict)
        == "Test: 97% 29/30 [00:50<00:00, 10.00it/s, foo=bar]"
    )
