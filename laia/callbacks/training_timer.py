import datetime

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import laia.common.logging as log
from laia.callbacks.meters import Timer

_logger = log.get_logger(__name__)


class TrainingTimer(pl.Callback):
    def __init__(self):
        super().__init__()
        self.tr_timer = Timer()
        self.va_timer = Timer()

    @staticmethod
    def time_to_str(time: float) -> str:
        return str(datetime.timedelta(seconds=time))

    @rank_zero_only
    def on_train_epoch_start(self, *args, **kwargs):
        super().on_train_epoch_start(*args, **kwargs)
        self.tr_timer.reset()

    @rank_zero_only
    def on_validation_epoch_start(self, *args, **kwargs):
        super().on_validation_epoch_start(*args, **kwargs)
        self.va_timer.reset()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, *args, **kwargs):
        super().on_train_epoch_end(trainer, *args, **kwargs)
        _logger.info(
            f"E{trainer.current_epoch}: "
            f"tr_time={self.time_to_str(self.tr_timer.value)}, "
            f"va_time={self.time_to_str(self.va_timer.value)}"
        )
