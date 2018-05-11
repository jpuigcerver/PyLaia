import sys

PY3 = sys.version_info[0] == 3


class EngineException(Exception):
    def __init__(self, epoch, iteration, batch, cause=None):
        self._epoch = epoch
        self._iteration = iteration
        self._batch = batch
        self._cause = cause

    def __str__(self):
        if not self._cause:
            msg = ('Exception raised during epoch {}, iteration {}. '
                   'The batch that caused the exception was: {}'.format(
                       self._epoch, self._iteration, self._batch))
        else:
            msg = ('Exception "{!r}" raised during epoch {}, iteration {}. '
                   'The batch that caused the exception was: {}'.format(
                       self._cause, self._epoch, self._iteration, self._batch))
        return msg
