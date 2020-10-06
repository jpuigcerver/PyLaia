import datetime

import pytorch_lightning as pl

import laia.common.logging as log
from laia.callbacks.meters import Timer

_logger = log.get_logger(__name__)


class TrainingTimer(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.tr_timer = Timer()
        self.va_timer = Timer()

    @staticmethod
    def time_to_str(time: float) -> str:
        return str(datetime.timedelta(seconds=time))

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.tr_timer.reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        self.va_timer.reset()

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        _logger.info(
            f"Epoch {trainer.current_epoch}: "
            f"tr_time={self.time_to_str(self.tr_timer.value)}, "
            f"va_time={self.time_to_str(self.va_timer.value)}"
        )
