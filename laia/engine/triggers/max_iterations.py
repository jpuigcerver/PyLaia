from __future__ import absolute_import

from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import Trigger


class MaxIterations(Trigger):
    r"""Trigger after a given number of iterations."""
    def __init__(self, trainer, max_iterations):
        assert isinstance(trainer, Trainer)
        self._trainer = trainer
        self._max_iterations = max_iterations

    def __call__(self):
        if self._trainer.iterations >= self._max_iterations:
            return True
        else:
            return False
