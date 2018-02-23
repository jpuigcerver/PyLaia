from __future__ import absolute_import

from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import Trigger


class MaxEpochs(Trigger):
    r"""Trigger after a given number of epochs."""
    def __init__(self, trainer, max_epochs):
        assert isinstance(trainer, Trainer)
        self._trainer = trainer
        self._max_epochs = max_epochs

    def __call__(self):
        if self._trainer.epochs >= self._max_epochs:
            return True
        else:
            return False
