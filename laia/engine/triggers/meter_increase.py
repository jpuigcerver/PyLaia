from __future__ import absolute_import

import logging

from laia.engine.triggers.trigger import TriggerLogWrapper
from laia.engine.triggers.trigger_from_meter import TriggerFromMeter
from laia.meters.meter import Meter

from typing import Any

_logger = logging.getLogger(__name__)


class MeterIncrease(TriggerFromMeter):
    def __init__(self, meter, meter_key=None, name=None):
        # type: (Meter, Any, str) -> None
        super(MeterIncrease, self).__init__(meter, _logger, meter_key, name)
        self._highest = float('-inf')

    def _process_value(self, last_value):
        if last_value > self._highest:
            self.logger.info(
                TriggerLogWrapper(
                    self, 'New highest value {} (previous was {})',
                    last_value, self._highest))
            self._highest = last_value
            return True
        else:
            self.logger.debug(
                TriggerLogWrapper(
                    self, 'Value IS NOT the highest (last: {} vs highest: {})',
                    last_value, self._highest))
            return False
