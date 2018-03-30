from __future__ import absolute_import

from laia.engine.conditions.condition import Condition


class MultipleOf(Condition):
    """True if the dividend is a multiple of the divisor.

    Arguments:
        dividend (int)
        divisor (int)
    """

    def __init__(self, dividend, divisor):
        # type: (int, int) -> None
        assert divisor > 0
        super(MultipleOf, self).__init__()
        self._dividend = dividend
        self._divisor = divisor

    def __call__(self):
        return self._dividend % self._divisor == 0
