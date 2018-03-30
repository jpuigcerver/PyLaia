from __future__ import absolute_import

from laia.engine.conditions.condition import Condition


class TargetReached(Condition):
    """True if the target is reached.

    Arguments:
        target (int): number to reach
        current (int): current number
    """

    def __init__(self, target, current):
        # type: (int, int) -> None
        super(TargetReached, self).__init__()
        self._target = target
        self._current = current

    def __call__(self):
        return self._target <= self._current
