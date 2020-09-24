import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import laia.common.logging as log

_logger = log.get_logger(__name__)


class LearningRate(pl.callbacks.LearningRateMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_values = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.last_values = {}

    def on_epoch_end(self, trainer, pl_module):
        super().on_epoch_end(trainer, pl_module)
        for k, v in self.lrs.items():
            prev_value = self.last_values.get(k, None)
            new_value = v[-1]
            if prev_value is not None and prev_value != new_value:
                _logger.info("{} modified: {} ⟶ {}", k, prev_value, new_value)
            self.last_values[k] = new_value
