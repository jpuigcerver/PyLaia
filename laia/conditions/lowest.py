from __future__ import absolute_import

from typing import Callable, Any

import numpy as np

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class Lowest(LoggingCondition):
    """True if a new lowest value has been reached"""

    def __init__(self, obj, key=None, name=None):
        # type: (Callable, Any, str) -> None
        super(Lowest, self).__init__(obj, key, _logger, name)
        self._lowest = np.inf

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        if value < self._lowest:
            self.info("New lowest value {} (previous was {})", value, self._lowest)
            self._lowest = value
            return True
        self.debug(
            "Value IS NOT the lowest (last: {} vs lowest: {})", value, self._lowest
        )
        return False

    def state_dict(self):
        state = super(Lowest, self).state_dict()
        state["lowest"] = self._lowest
        return state

    def load_state_dict(self, state):
        super(Lowest, self).load_state_dict(state)
        self._lowest = state["lowest"]
