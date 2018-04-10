from __future__ import absolute_import

from typing import Callable

import numpy as np

import laia.logging as log
from laia.hooks.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class NonIncreasing(LoggingCondition):
    def __init__(self, obj, max_no_increasing_calls, key=None, name=None):
        # type: (Callable, int, Any, str) -> None
        super(NonIncreasing, self).__init__(obj, key, _logger, name)
        self._highest = np.NINF
        self._max_no_increase = max_no_increasing_calls
        self._highest_call = 0
        self._calls = 0

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        self._calls += 1
        if value > self._highest:
            self.info('New highest value {} (previous was {})',
                      value, self._highest)
            self._highest = value
            self._highest_call += 1
            return False
        elif self._calls - self._highest_call >= self._max_no_increase:
            if self._calls - self._highest_call == self._max_no_increase:
                self.info('Highest value {} DID NOT increase after {} calls',
                          self._highest, self._max_no_increase)
            return True
        else:
            return False
