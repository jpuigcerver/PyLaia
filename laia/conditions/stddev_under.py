from __future__ import absolute_import

from typing import Callable, Any

import numpy as np

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class StdDevUnder(LoggingCondition):
    r"""Returns True when the standard deviation
    over the latest values is lower than some threshold.

    Each time this is called with a value, it will be stored.

    The condition will only return True when the standard deviation over the
    last ``num_values_to_keep``  values received is below
    the given ``threshold``.

    Arguments:
        threshold (float): the standard deviation threshold used to return
            True.
        num_values_to_keep (int): the amount of values to keep track of.
        name (str): Name for the condition.
    """

    def __init__(self, obj, threshold, num_values_to_keep, key=None, name=None):
        # type: (Callable, float, int, Any, str) -> None
        assert threshold > 0, "Standard deviation should be a positive value"
        assert num_values_to_keep > 1, (
            "The number of values to keep must be greater than 1 to compute "
            "the standard deviation"
        )
        super(StdDevUnder, self).__init__(obj, key, _logger, name)
        self._threshold = threshold
        self._num_values_to_keep = num_values_to_keep
        self._values = []
        self._nval = 0

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        # Add last_value to the values
        if self._num_values_to_keep > len(self._values):
            self._values.append(value)
        else:
            self._values[self._nval] = value
            self._nval = (self._nval + 1) % self._num_values_to_keep

        # If not enough values are kept, return False
        if len(self._values) < self._num_values_to_keep:
            return False

        std = np.std(np.asarray(self._values, dtype=np.float32))
        if std < self._threshold:
            self.info("Standard deviation {} < Threshold {}", std, self._threshold)
            return True
        self.debug("Standard deviation {} >= Threshold {}", std, self._threshold)
        return False
