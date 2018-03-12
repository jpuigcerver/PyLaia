from __future__ import absolute_import

import laia.plugins.logging as log
from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import TriggerLogWrapper


class NumIterations():
    """Trigger after the given `trainer` reaches a given number of iterations.

    Arguments:
        trainer (:obj:`~laia.engine.Trainer`) : trainer to monitor
        num_iterations (int) : number of iterations to reach
        name (str) : name of the trigger
    """

    def __init__(self, trainer, num_iterations, name=None):
        # type: (Trainer, int, str) -> None
        assert isinstance(trainer, Trainer)
        self._trainer = trainer
        self._num_iterations = num_iterations
        self._logger = log.get_logger(name)

    def __call__(self):
        if self._trainer.iterations >= self._num_iterations:
            self._logger.info(TriggerLogWrapper(
                self, 'Trainer reached {} iterations',
                self._num_iterations))
            return True
        else:
            self._logger.debug(TriggerLogWrapper(
                self, 'Trainer DID NOT reach {} iterations yet',
                self._num_iterations))
            return False
