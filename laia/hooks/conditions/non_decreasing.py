from __future__ import absolute_import

from typing import Callable, Any

import numpy as np

import laia.logging as log
from laia.hooks.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class NonDecreasing(LoggingCondition):
    def __init__(self, obj, max_no_decreasing_calls, key=None, name=None):
        # type: (Callable, int, Any, str) -> None
        super(NonDecreasing, self).__init__(obj, key, _logger, name)
        self._lowest = np.inf
        self._max_no_decrease = max_no_decreasing_calls
        self._lowest_call = 0
        self._calls = 0

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        self._calls += 1
        if value < self._lowest:
            self.info('New lowest value {} (previous was {})',
                      value, self._lowest)
            self._lowest = value
            self._lowest_call += 1
            return True
        elif self._calls - self._lowest_call >= self._max_no_decrease:
            if self._calls - self._lowest_call == self._max_no_decrease:
                self.info('Lowest value {} DID NOT decrease after {} calls',
                          self._lowest, self._max_no_decrease)
            return True
        else:
            return False
