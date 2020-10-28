import sys
from collections import defaultdict
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import convert_inf
from tqdm.auto import tqdm

import laia.common.logging as log
from laia.callbacks.meters import Timer

_logger = log.get_logger(__name__)


class ProgressBar(pl.callbacks.ProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        ncols: Optional[int] = 120,
        dynamic_ncols: bool = False,
    ):
        super().__init__(refresh_rate=refresh_rate)
        self.ncols = ncols
        self.dynamic_ncols = dynamic_ncols
        self.running_sanity = None
        self.level = log.INFO
        self.format = defaultdict(
            self.format_factory,
            {
                "loss": "{:.4f}",
                "cer": "{:.1%}",
                "wer": "{:.1%}",
            },
        )
        # lightning merges tr + va into a "main bar".
        # we want to keep it separate so we have to time it ourselves
        self.tr_timer = Timer()
        self.va_timer = Timer()

    @staticmethod
    def format_factory():
        return "{}"

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

    def on_epoch_start(self, trainer, *args, **kwargs):
        # skip parent
        super(pl.callbacks.ProgressBar, self).on_epoch_start(trainer, *args, **kwargs)
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(convert_inf(self.total_train_batches))
        self.main_progress_bar.set_description(f"TR - Epoch {trainer.current_epoch}")

    def on_train_epoch_start(self, *args, **kwargs):
        super().on_train_epoch_start(*args, **kwargs)
        self.tr_timer.reset()

    def on_validation_epoch_start(self, trainer, *args, **kwargs):
        super().on_validation_start(trainer, *args, **kwargs)
        if trainer.running_sanity_check:
            self.val_progress_bar.set_description("VA sanity check")
        else:
            self.tr_timer.stop()
            self.va_timer.reset()
            self.val_progress_bar.set_description(f"VA - Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # skip parent to avoid two postfix calls
        super(pl.callbacks.ProgressBar, self).on_train_batch_end(
            trainer, *args, **kwargs
        )
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.set_postfix(
                refresh=True,
                running_loss=trainer.progress_bar_dict["loss"],
                **trainer.progress_bar_metrics.get("gpu_stats", {}),
            )

    def set_postfix(self, pbar, prefix):
        l = len(prefix)
        postfix = {
            k[l:]: self.format[k[l:]].format(v)
            for k, v in self.trainer.progress_bar_dict.items()
            if k.startswith(prefix)
        }
        pbar.set_postfix(postfix, refresh=True)

    @staticmethod
    def fix_format_dict_time(pbar, prefix, timer):
        format_dict = pbar.format_dict
        _logger.debug(
            f"{prefix} - lightning-elapsed={format_dict['elapsed']} elapsed={timer.value}"
        )
        format_dict["elapsed"] = timer.value
        return format_dict

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        super().on_train_epoch_end(trainer, *args, **kwargs)
        if self.is_enabled:
            # add metrics to training bar
            self.set_postfix(self.main_progress_bar, "tr_")
            # override training time
            format_dict = self.fix_format_dict_time(
                self.main_progress_bar, "TR", self.tr_timer
            )
            # log training bar
            _logger.log(self.level, tqdm.format_meter(**format_dict))

            if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch:
                return
            # add metrics to training bar.
            # note: this is here instead of in `on_validation_epoch_end`
            # because `val_loop` gets called before `on_train_epoch_end` so the VA
            # bar would get printed before the TR bar. see:
            # https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#hook-lifecycle-pseudocode
            self.set_postfix(self.val_progress_bar, "va_")
            # override validation time
            format_dict = self.fix_format_dict_time(
                self.val_progress_bar, "VA", self.va_timer
            )
            # log validation bar
            _logger.log(self.level, tqdm.format_meter(**format_dict))

    def on_validation_batch_end(self, *args, **kwargs):
        # skip parent
        super(pl.callbacks.ProgressBar, self).on_validation_batch_end(*args, **kwargs)
        if self.is_enabled and self.val_batch_idx % self.refresh_rate == 0:
            self.val_progress_bar.update(self.refresh_rate)

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        super().on_validation_epoch_end(trainer, *args, **kwargs)
        if self.is_enabled:
            if trainer.running_sanity_check:
                self.val_progress_bar.refresh()
                _logger.log(self.level, str(self.val_progress_bar))
            else:
                self.va_timer.stop()

    def on_validation_end(self, *args, **kwargs):
        # skip parent to avoid postfix call
        super(pl.callbacks.ProgressBar, self).on_validation_end(*args, **kwargs)
        self.val_progress_bar.close()

    def on_test_end(self, *args, **kwargs):
        self.test_progress_bar.clear()
        super().on_test_end(*args, **kwargs)
