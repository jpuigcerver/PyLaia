from __future__ import absolute_import

import logging

from laia.engine.triggers.trigger import LoggedTrigger, TriggerLogWrapper
from laia.meters.meter import Meter

from typing import Any

_logger = logging.getLogger(__name__)


class MeterDecrease(LoggedTrigger):
    def __init__(self, meter, meter_key=None, name=None):
        # type: (Meter, Any, str) -> None
        assert isinstance(meter, Meter)
        super(MeterDecrease, self).__init__(_logger, name)
        self._meter = meter
        self._meter_key = meter_key
        self._lowest = float('inf')

    def __call__(self):
        # Try to get the meter's last value, if some exception occurs,
        # we assume that the meter has not produced any value yet, and
        # we do not trigger.
        try:
            last_value = self._meter.value
            if last_value is None:
                raise TypeError('Meter returned None')
        except Exception:
            self.logger.exception(
                TriggerLogWrapper(self, 'No value fetched from meter'))
            return False

        if self._meter_key is not None:
            last_value = last_value[self._meter_key]

        # Return True, iff the value read from the meter is lower than the
        # lowest value seen so far.
        if last_value < self._lowest:
            self.logger.info(
                TriggerLogWrapper(
                    self, 'New lowest value {} (previous was {})',
                    last_value, self._lowest))
            self._lowest = last_value
            return True
        else:
            self.logger.debug(
                TriggerLogWrapper(
                    self, 'Value IS NOT the lowest (last: {} vs lowest: {})',
                    last_value, self._lowest))
            return False
