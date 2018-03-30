from __future__ import absolute_import

from typing import Any

import numpy as np

from laia.engine.conditions.condition import ConditionFromMeter
from laia.hooks.meters import Meter


class NonDecreasing(ConditionFromMeter):
    def __init__(self, meter, max_no_decreasing_calls,
                 meter_key=None, name=None):
        # type: (Meter, int, Any, str) -> None
        super(NonDecreasing, self).__init__(meter, meter_key, name)
        self._lowest = np.inf
        self._max_no_decrease = max_no_decreasing_calls
        self._lowest_call = 0
        self._calls = 0

    def _process_value(self, last_value):
        self._calls += 1
        if last_value < self._lowest:
            self.logger.info(
                'New lowest value {} (previous was {})',
                last_value, self._lowest)
            self._lowest = last_value
            self._lowest_call += 1
            return True
        elif self._calls - self._lowest_call >= self._max_no_decrease:
            if self._calls - self._lowest_call == self._max_no_decrease:
                self.logger.info(
                    'Lowest value {} DID NOT decrease after {} calls',
                    self._lowest, self._max_no_decrease)
            return True
        else:
            return False
