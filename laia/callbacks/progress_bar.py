import sys

import pytorch_lightning as pl
from tqdm.auto import tqdm

import laia.common.logging as log

_logger = log.get_logger(__name__)


class ProgressBar(pl.callbacks.ProgressBar):
    def __init__(self, refresh_rate: int = 1):
        super().__init__(refresh_rate=refresh_rate)
        self.ncols = 120
        self.dynamic_ncols = False
        self.running_sanity = None
        self.level = log.INFO
        self.format = {
            "loss": "{:.4f}",
            "cer": "{:.1%}",
            "wer": "{:.1%}",
        }

    def init_sanity_tqdm(self) -> tqdm:
        return tqdm(
            desc="VA sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
        )

    def init_train_tqdm(self) -> tqdm:
        return tqdm(
            desc="TR",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
            smoothing=0,
        )

    def init_validation_tqdm(self) -> tqdm:
        return tqdm(
            desc="VA",
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
        )

    def init_test_tqdm(self) -> tqdm:
        return tqdm(
            desc="Decoding",
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=True,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
        )

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"TR - Epoch {trainer.current_epoch}")

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.val_progress_bar.set_description(
            "VA sanity check"
            if trainer.running_sanity_check
            else f"VA - Epoch {trainer.current_epoch}"
        )

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # skip parent to avoid two postfix calls
        super(pl.callbacks.ProgressBar, self).on_train_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.set_postfix(
                running_loss=trainer.progress_bar_dict["loss"], refresh=True
            )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.is_enabled:
            # TODO: remove if https://github.com/PyTorchLightning/pytorch-lightning/issues/3231
            # log training bar
            postfix = {
                k[3:]: self.format[k[3:]].format(v)
                for k, v in self.trainer.progress_bar_dict.items()
                if k.startswith("tr_")
            }
            self.main_progress_bar.set_postfix(postfix, refresh=True)
            _logger.log(self.level, str(self.main_progress_bar))

            # log validation bar. this is here instead of in `on_validation_epoch_end`
            # because `val_loop` gets called before `on_train_epoch_end` so the VA
            # bar would get printed before the TR bar. see:
            # https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#hook-lifecycle-pseudocode
            postfix = {
                k[3:]: self.format[k[3:]].format(v)
                for k, v in self.trainer.progress_bar_dict.items()
                if k.startswith("va_")
            }
            self.val_progress_bar.set_postfix(postfix, refresh=True)
            _logger.log(self.level, str(self.val_progress_bar))

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self.is_enabled and trainer.running_sanity_check:
            # only when running sanity
            self.val_progress_bar.refresh()
            _logger.log(self.level, str(self.val_progress_bar))

    def on_validation_end(self, trainer, pl_module):
        # skip parent to avoid postfix call
        super(pl.callbacks.ProgressBar, self).on_validation_end(trainer, pl_module)
        self.val_progress_bar.close()

    def on_test_end(self, trainer, pl_module):
        self.test_progress_bar.clear()
        super().on_test_end(trainer, pl_module)
