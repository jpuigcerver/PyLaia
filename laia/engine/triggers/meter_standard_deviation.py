from __future__ import absolute_import

import logging

import numpy as np
from typing import Any

from laia.engine.triggers.trigger import LoggedTrigger, TriggerLogWrapper
from laia.meters.meter import Meter

_logger = logging.getLogger(__name__)


class MeterStandardDeviation(LoggedTrigger):
    r"""This trigger returns True when the standard deviation of a given meter
    over the latest values is lower than some threshold.

    Each time the Trigger is called, it will read the value of the ``meter`` and
    will store it.

    The trigger will only return True when the standard deviation over the
    last ``num_values_to_keep``  values read from the ``meter`` is below
    the given ``threshold``.

    Arguments:
    meter (:obj:`laia.meters.Meter`): the meter whose value will be
        monitored.
    threshold (float): the standard deviation threshold used to trigger
        True.
    num_values_to_keep (int): the size of the values over the meter
        values.
    meter_key (any): If given, get this key from the `meter` value.
        Useful when the `meter` value is a tuple/list/dict. (default: None)
    name (str): Name for the trigger.
    """

    def __init__(self, meter, threshold, num_values_to_keep, meter_key=None,
                 name=None):
        # type: (Meter, float, int, Any, str) -> None
        assert isinstance(meter, Meter)
        assert threshold > 0, 'Standard deviation should be a positive value'
        assert num_values_to_keep > 1, (
            'The number of values to keep must be greater than 1 to compute '
            'the standard deviation')
        super(MeterStandardDeviation, self).__init__(_logger, name)
        self._meter = meter
        self._meter_key = meter_key
        self._threshold = threshold
        self._num_values_to_keep = num_values_to_keep
        self._values = []
        self._nval = 0

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

        # Add last_value to the values
        if self._num_values_to_keep > len(self._values):
            self._values.append(last_value)
        else:
            self._values[self._nval] = last_value
            self._nval = (self._nval + 1) % self._num_values_to_keep

        # If not enough values are kept, return False
        if len(self._values) < self._num_values_to_keep:
            return False

        std = np.std(np.asarray(self._values, dtype=np.float32))
        if std < self._threshold:
            self.logger.info(
                TriggerLogWrapper(
                    self, 'Standard deviation {} < Theshold {}',
                    std, self._threshold))
            return True
        else:
            self.logger.debug(
                TriggerLogWrapper(
                    self, 'Standard deviation {} >= Theshold {}',
                    std, self._threshold))
            return False
