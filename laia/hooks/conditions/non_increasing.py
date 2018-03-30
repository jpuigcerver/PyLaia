from __future__ import absolute_import

from typing import Any

import numpy as np

from laia.engine.conditions.condition import ConditionFromMeter
from laia.hooks.meters import Meter


class NonIncreasing(ConditionFromMeter):
    def __init__(self, meter, max_no_increasing_calls,
                 meter_key=None, name=None):
        # type: (Meter, int, Any, str) -> None
        super(NonIncreasing, self).__init__(meter, meter_key, name)
        self._highest = np.NINF
        self._max_no_increase = max_no_increasing_calls
        self._highest_call = 0
        self._calls = 0

    def _process_value(self, last_value):
        self._calls += 1
        if last_value > self._highest:
            self.logger.info(
                'New highest value {} (previous was {})',
                last_value, self._highest)
            self._highest = last_value
            self._highest_call += 1
            return False
        elif self._calls - self._highest_call >= self._max_no_increase:
            if self._calls - self._highest_call == self._max_no_increase:
                self.logger.info(
                    'Highest value {} DID NOT increase after {} calls',
                    self._highest, self._max_no_increase)
            return True
        else:
            return False
