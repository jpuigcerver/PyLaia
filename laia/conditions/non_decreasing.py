from __future__ import absolute_import

from typing import Callable, Any

from torch._six import inf

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class NonDecreasing(LoggingCondition):
    """Returns True after it has  been called `max_no_decreasing_calls`
    times without a new lowest value"""

    def __init__(self, obj, max_no_decreasing_calls, key=None, name=None):
        # type: (Callable, int, Any, str) -> None
        super(NonDecreasing, self).__init__(obj, key, _logger, name)
        self._lowest = inf
        self._max_no_decrease = max_no_decreasing_calls
        self._lowest_calls = 0
        self._calls = 0

    def __call__(self):
        self._calls += 1
        if self._calls - self._lowest_calls >= self._max_no_decrease:
            if self._calls - self._lowest_calls == self._max_no_decrease:
                self.info(
                    "Lowest value {} DID NOT decrease after {} calls",
                    self._lowest,
                    self._max_no_decrease,
                )
            return True
        value = self._process_value()
        if value is not None and value < self._lowest:
            self.info("New lowest value {} (previous was {})", value, self._lowest)
            self._lowest = value
            self._lowest_calls += 1
        return False

    def state_dict(self):
        state = super(NonDecreasing, self).state_dict()
        state["lowest"] = self._lowest
        state["lowest_calls"] = self._lowest_calls
        state["calls"] = self._calls
        return state

    def load_state_dict(self, state):
        super(NonDecreasing, self).load_state_dict(state)
        self._lowest = state["lowest"]
        self._lowest_calls = state["lowest_calls"]
        self._calls = state["calls"]


class ConsecutiveNonDecreasing(LoggingCondition):
    """Returns True after it has  been called `max_no_decreasing_calls`
    consecutive times without a new lowest value"""

    def __init__(self, obj, max_no_decreasing_calls, key=None, name=None):
        # type: (Callable, int, Any, str) -> None
        super(ConsecutiveNonDecreasing, self).__init__(obj, key, _logger, name)
        self._lowest = inf
        self._max_no_decrease = max_no_decreasing_calls
        self._calls = 0

    def __call__(self):
        self._calls += 1
        if self._calls >= self._max_no_decrease:
            if self._calls == self._max_no_decrease:
                self.info(
                    "Lowest value {} DID NOT decrease after {} consecutive calls",
                    self._lowest,
                    self._max_no_decrease,
                )
            return True
        value = self._process_value()
        if value is not None and value < self._lowest:
            self.info("New lowest value {} (previous was {})", value, self._lowest)
            self._lowest = value
            self._calls = 0
        return False

    def state_dict(self):
        state = super(ConsecutiveNonDecreasing, self).state_dict()
        state["lowest"] = self._lowest
        state["calls"] = self._calls
        return state

    def load_state_dict(self, state):
        super(ConsecutiveNonDecreasing, self).load_state_dict(state)
        self._lowest = state["lowest"]
        self._calls = state["calls"]
