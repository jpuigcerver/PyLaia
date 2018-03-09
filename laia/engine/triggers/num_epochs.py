from __future__ import absolute_import

import laia.plugins.logging as log
from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import TriggerLogWrapper


class NumEpochs():
    """Trigger after the given `trainer` reaches a given number of epochs.

    Arguments:
        trainer (:obj:`~laia.engine.Trainer`) : trainer to monitor
        num_epochs (int) : number of epochs to reach
        name (str) : name of the trigger
    """

    def __init__(self, trainer, num_epochs, name=None):
        # type: (Trainer, int, str) -> None
        assert isinstance(trainer, Trainer)
        super(NumEpochs, self).__init__(name)
        self._trainer = trainer
        self._num_epochs = num_epochs

    def __call__(self):
        if self._trainer.epochs >= self._num_epochs:
            log.info(
                TriggerLogWrapper(
                    self, 'Trainer reached {} epochs',
                    self._num_epochs))
            return True
        else:
            log.debug(TriggerLogWrapper(
                self, 'Trainer DID NOT reach {} epochs yet',
                self._num_epochs))
            return False
