from __future__ import absolute_import

import numpy as np

from laia.engine.triggers.trigger import Trigger
from laia.meters.meter import Meter
from laia.meters.running_average_meter import RunningAverageMeter


class MeterStandardDeviation(Trigger):
    r"""This trigger returns True when the standard deviation of a given meter
    over the latest values is lower than some threshold.

    Each time the Trigger is called, it will read the value of the ``meter`` and
    will store it.

    The trigger will only return True when the standard deviation over the
    last ``num_values_to_keep``  values read from the ``meter`` is below
    the given ``threshold``.
    """

    def __init__(self, meter, threshold, num_values_to_keep):
        r"""Creates a new MeterStandardDeviation trigger.

        The trigger will only return True when the standard deviation over the
        last ``num_values_to_keep``  values read from the ``meter`` is below
        the given ``threshold``.

        Args:
          meter (laia.meters.Meter): the meter whose value will be monitored.
          threshold (float): the standard deviation threshold used to trigger
            True.
          num_values_to_keep (int): the size of the values over the meter
            values.
        """
        assert isinstance(meter, Meter)
        assert threshold > 0, ('Standard deviation should be a positive value')
        assert num_values_to_keep > 1, (
            'The number of values to keep must be greater than 1 to compute '
            'the standard deviation')
        self._meter = meter
        self._threshold = threshold
        self._num_values_to_keep = num_values_to_keep
        self._values = []
        self._nval = 0

    def __call__(self):
        # Try to get the meter's last value, if some exception occurrs,
        # we assume that the meter has not produced any value yet, and
        # we do not trigger.
        try:
            last_value = self._meter.value
        except Exception:
            # TODO(jpuigcerver): We might want to log this, just in case the
            # user has made some dummy mistake that prevents this trigger
            # from returning True.
            return False

        # Note: RunningAverageMeter returns a tuple, with the current mean
        # and standard deviation, use only the mean.
        if isinstance(self._meter, RunningAverageMeter):
            last_value = last_value[0]

        # Add last_value to the values
        if self._num_values_to_keep > len(self._values):
            self._values.append(last_value)
        else:
            self._values[self._nval] = last_value
            self._nval = (self._nval + 1) % self._num_values_to_keep

        # If not enough values are kept, return False
        if len(self._values) < self._num_values_to_keep:
            return False

        if np.std(np.asarray(self._values, dtype=np.float32)) < self._threshold:
            return True
        else:
            return False
