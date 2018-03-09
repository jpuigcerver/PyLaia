from __future__ import absolute_import

import laia.plugins.logging as log
from laia.engine.triggers.trigger import TriggerLogWrapper
from laia.meters.meter import Meter


class TriggerFromMeter():
    """Base class for triggers based on the value of a :class:`Meter`.

    If the value of the meter cannot be read (`Meter.value` returns `None` or
    causes an exception), we assume that the meter has not produced a value
    yet and the trigger returns false.

    The number of exceptions will be recorded and an error will be logged
    when `num_exceptions_threshold` are caught.

    Arguments:
        meter (:obj:`Meter`): meter to monitor.
        meter_key (Any): if the value returned by the meter is a tuple, list
            or dictionary, use this key to get the specific value.
            (default: None)
        name (str): name for the logger (default: None).
        num_exceptions_threshold (int): number of exceptions to catch from the
            meter before logging. (default: 5)
    """

    def __init__(self, meter, meter_key=None, name=None,
                 num_exceptions_threshold=5):
        # type: (Meter, str) -> None
        self._meter = meter
        self._meter_key = meter_key
        self._num_exceptions = 0
        self._name = name
        self._num_exceptions_threshold = num_exceptions_threshold

    @property
    def meter(self):
        return self._meter

    def _process_value(self, last_value):
        raise NotImplementedError('This method should be implemented')

    def __call__(self):
        # Try to get the meter's last value, if some exception occurs,
        # we assume that the meter has not produced any value yet, and
        # we do not trigger.
        try:
            last_value = self._meter.value
            if last_value is None:
                raise TypeError('Meter returned None')
        except Exception:
            self._num_exceptions += 1
            if self._num_exceptions % self._num_exceptions_threshold == 0:
                log.warn(TriggerLogWrapper(
                    self,
                    'No value fetched from meter after a while '
                    '({} exceptions like this occured so far)',
                    self._num_exceptions), name=self._name)
            return False

        if self._meter_key is not None:
            last_value = last_value[self._meter_key]

        return self._process_value(last_value)
