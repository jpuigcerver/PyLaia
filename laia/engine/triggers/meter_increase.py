from __future__ import absolute_import

from laia.engine.triggers.trigger import Trigger
from laia.meters.meter import Meter
from laia.meters.running_average_meter import RunningAverageMeter


class MeterIncrease(Trigger):
    def __init__(self, meter):
        assert isinstance(meter, Meter)
        self._meter = meter
        self._highest = float('-inf')

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

        # Return True, iff the value read from the meter is higher than the
        # highest value seen so far.
        if last_value > self._highest:
            self._highest = last_value
            return True
        else:
            return False
