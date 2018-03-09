from __future__ import absolute_import

import laia.plugins.logging as log
from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import TriggerLogWrapper


class NumUpdates():
    """Trigger after the given `trainer` reaches a given number of updates.

    Arguments:
        trainer (:obj:`~laia.engine.Trainer`) : trainer to monitor
        num_updates (int) : number of updates to reach
        name (str) : name of the trigger
    """

    def __init__(self, trainer, num_updates, name=None):
        # type: (Trainer, int, str) -> None
        assert isinstance(trainer, Trainer)
        super(NumUpdates, self).__init__(name)
        self._trainer = trainer
        self._num_updates = num_updates

    def __call__(self):
        if self._trainer.updates >= self._num_updates:
            log.info(TriggerLogWrapper(self, 'Trainer reached {} updates',
                                       self._num_updates), name=__name__)
            return True
        else:
            log.debug(TriggerLogWrapper(self, 'Trainer DID NOT reach {} updates yet',
                                        self._num_updates), name=__name__)
            return False
