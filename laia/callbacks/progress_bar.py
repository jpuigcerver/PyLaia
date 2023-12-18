import sys
from collections import defaultdict
from logging import INFO
from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import convert_inf
from tqdm.auto import tqdm

import laia.common.logging as log
from laia.callbacks.meters import Timer


class ProgressBar(pl.callbacks.ProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        ncols: Optional[int] = 120,
        dynamic_ncols: bool = True,
    ):
        super().__init__(refresh_rate=refresh_rate)
        self.ncols = ncols
        self.dynamic_ncols = dynamic_ncols
        self.running_sanity = None
        self.level = INFO
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

    def on_train_epoch_start(self, trainer, pl_module, *args, **kwargs):
        super().on_train_epoch_start(trainer, pl_module, *args, **kwargs)
        self.tr_timer.reset()
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(convert_inf(self.total_train_batches))
        self.main_progress_bar.set_description_str(f"TR - E{trainer.current_epoch}")

    def on_validation_epoch_start(self, trainer, *args, **kwargs):
        super().on_validation_start(trainer, *args, **kwargs)
        if trainer.running_sanity_check:
            self.val_progress_bar.set_description_str("VA sanity check")
        else:
            self.tr_timer.stop()
            self.va_timer.reset()
            self.val_progress_bar.set_description_str(f"VA - E{trainer.current_epoch}")

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # skip parent to avoid two postfix calls
        super(pl.callbacks.ProgressBar, self).on_train_batch_end(
            trainer, *args, **kwargs
        )
        if self._should_update(self.train_batch_idx, self.total_train_batches):
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(
                refresh=True,
                running_loss=trainer.progress_bar_dict["loss"],
                **trainer.progress_bar_metrics.get("gpu_stats", {}),
            )

    def set_postfix(self, pbar, prefix):
        l = len(prefix)
        postfix = {}
        for k, v in self.trainer.progress_bar_dict.items():
            if k.startswith(prefix):
                postfix[k[l:]] = self.format[k[l:]].format(v)
            elif not k.startswith("tr_") and not k.startswith("va_"):
                postfix[k] = v
        pbar.set_postfix(postfix, refresh=True)

    @staticmethod
    def fix_format_dict(
        pbar: tqdm, prefix: Optional[str] = None, timer: Optional[Timer] = None
    ) -> Dict:
        format_dict = pbar.format_dict
        if timer is not None:
            log.debug(
                f"{prefix} - lightning-elapsed={format_dict['elapsed']} elapsed={timer.value}"
            )
            format_dict["elapsed"] = timer.value
        # remove the square blocks, they provide no info
        format_dict["bar_format"] = (
            "{desc}: {percentage:.0f}% {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_noinv:.2f}it/s{postfix}]"
        )
        return format_dict

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        super().on_train_epoch_end(trainer, *args, **kwargs)
        if self.is_enabled:
            # add metrics to training bar
            self.set_postfix(self.main_progress_bar, "tr_")
            # override training time
            format_dict = self.fix_format_dict(
                self.main_progress_bar, "TR", self.tr_timer
            )
            # log training bar
            log.log(self.level, tqdm.format_meter(**format_dict))

    def on_validation_batch_end(self, *args, **kwargs):
        # skip parent
        super(pl.callbacks.ProgressBar, self).on_validation_batch_end(*args, **kwargs)
        if self._should_update(self.val_batch_idx, self.total_val_batches):
            self._update_bar(self.val_progress_bar)

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        super().on_validation_epoch_end(trainer, *args, **kwargs)

        if self.is_enabled:
            self.set_postfix(self.val_progress_bar, "va_")
            # override validation time
            format_dict = self.fix_format_dict(
                self.val_progress_bar, "VA", self.va_timer
            )
            # log validation bar
            log.log(self.level, tqdm.format_meter(**format_dict))

            if trainer.running_sanity_check:
                self.val_progress_bar.refresh()
                log.log(
                    self.level,
                    tqdm.format_meter(**self.fix_format_dict(self.val_progress_bar)),
                )
            else:
                self.va_timer.stop()

    def on_validation_end(self, *args, **kwargs):
        # skip parent to avoid postfix call
        super(pl.callbacks.ProgressBar, self).on_validation_end(*args, **kwargs)
        self.val_progress_bar.close()

    def on_test_end(self, *args, **kwargs):
        self.test_progress_bar.clear()
        super().on_test_end(*args, **kwargs)
