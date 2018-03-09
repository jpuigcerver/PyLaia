from __future__ import absolute_import

from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import Trigger


class EveryIteration(Trigger):
    def __init__(self, trainer, on_every_n):
        # type: (Trainer, int) -> None
        assert isinstance(trainer, Trainer)
        assert on_every_n > 0
        super(EveryIteration, self).__init__()
        self._trainer = trainer
        self._on_every_n = on_every_n

    def __call__(self):
        if self._trainer.iterations % self._on_every_n == 0:
            return True
        else:
            return False
