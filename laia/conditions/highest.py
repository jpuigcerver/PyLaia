from __future__ import absolute_import

from typing import Callable, Any

import numpy as np

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class Highest(LoggingCondition):
    """True if a new highest value has been reached"""

    def __init__(self, obj, key=None, name=None):
        # type: (Callable, Any, str) -> None
        super(Highest, self).__init__(obj, key, _logger, name)
        self._highest = np.NINF

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        if value > self._highest:
            self.info("New highest value {} (previous was {})", value, self._highest)
            self._highest = value
            return True
        self.debug(
            "Value IS NOT the highest (last: {} vs highest: {})", value, self._highest
        )
        return False

    def state_dict(self):
        state = super(Highest, self).state_dict()
        state["highest"] = self._highest
        return state

    def load_state_dict(self, state):
        super(Highest, self).load_state_dict(state)
        self._highest = state["highest"]
