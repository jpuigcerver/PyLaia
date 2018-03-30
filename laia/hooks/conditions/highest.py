from __future__ import absolute_import

from typing import Any

import numpy as np

from laia.engine.conditions.condition import ConditionFromMeter
from laia.hooks.meters import Meter


class Highest(ConditionFromMeter):
    """True each time a :class:`Meter` reaches a new highest value.

    Arguments:
        meter (:obj:`~laia.meters.Meter`): meter to monitor.
        meter_key: if the value returned by the meter is a tuple, list
            or dictionary, use this key to get the specific value.
            (default: None)
        name (str): name for the condition (default: None).
    """

    def __init__(self, meter, meter_key=None, name=None):
        # type: (Meter, Any, str) -> None
        super(Highest, self).__init__(meter, meter_key, name)
        self._highest = np.NINF

    def _process_value(self, last_value):
        if last_value > self._highest:
            self.logger.info(
                'New highest value {} (previous was {})',
                last_value, self._highest)
            self._highest = last_value
            return True
        else:
            self.logger.debug(
                'Value IS NOT the highest (last: {} vs highest: {})',
                last_value, self._highest)
            return False
