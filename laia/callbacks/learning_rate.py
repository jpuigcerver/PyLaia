import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import laia.common.logging as log

_logger = log.get_logger(__name__)


class LearningRate(pl.callbacks.LearningRateMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_values = None

    def on_train_start(self, trainer, *args, **kwargs):
        if not trainer.lr_schedulers:
            pl.utilities.rank_zero_warn(
                "You are using LearningRateMonitor callback with models "
                "that have no learning rate schedulers",
                RuntimeWarning,
            )
        names = self._find_names(trainer.lr_schedulers)
        self.lrs = {name: [] for name in names}
        self.last_values = {}

    @rank_zero_only
    def on_epoch_end(self, trainer, *args, **kwargs):
        super().on_epoch_end(trainer, *args, **kwargs)
        for k, v in self.lrs.items():
            prev_value = self.last_values.get(k, None)
            new_value = v[-1]
            if prev_value is not None and prev_value != new_value:
                _logger.info(
                    "Epoch {}: {} {:.3e} ⟶ {:.3e}",
                    trainer.current_epoch,
                    k,
                    prev_value,
                    new_value,
                )
            self.last_values[k] = new_value
