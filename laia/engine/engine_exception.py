from __future__ import absolute_import


class EngineException(Exception):
    def __init__(self, epoch, iteration, batch, cause=None):
        self._epoch = epoch
        self._iteration = iteration
        self._batch = batch
        self._cause = cause

    def __str__(self):
        return (
            "Exception {}raised during epoch {}, iteration {}. "
            "The batch that caused the exception was: {}".format(
                '"{!r}" '.format(self._cause) if self._cause else "",
                self._epoch,
                self._iteration,
                self._batch,
            )
        )
