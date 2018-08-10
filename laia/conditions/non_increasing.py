from typing import Callable, Any, Optional

from torch._six import inf

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class NonIncreasing(LoggingCondition):
    """Returns True after it has  been called `max_no_increasing_calls`
    times without a new highest value"""

    def __init__(
        self,
        obj: Callable,
        max_no_increasing_calls: int,
        key: Optional[Any] = None,
        name: str = None,
    ) -> None:
        super().__init__(obj, key, _logger, name)
        self._highest = -inf
        self._max_no_increase = max_no_increasing_calls
        self._highest_calls = 0
        self._calls = 0

    def __call__(self) -> bool:
        self._calls += 1
        if self._calls - self._highest_calls >= self._max_no_increase:
            if self._calls - self._highest_calls == self._max_no_increase:
                self.info(
                    "Highest value {} DID NOT increase after {} calls",
                    self._highest,
                    self._max_no_increase,
                )
            return True
        value = self._process_value()
        if value is not None and value < self._highest:
            self.info("New highest value {} (previous was {})", value, self._highest)
            self._highest = value
            self._highest_calls += 1
        return False

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["highest"] = self._highest
        state["highest_calls"] = self._highest_calls
        state["calls"] = self._calls
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._highest = state["highest"]
        self._highest_calls = state["highest_calls"]
        self._calls = state["calls"]


class ConsecutiveNonIncreasing(LoggingCondition):
    """Returns True after it has  been called `max_no_increasing_calls`
    consecutive times without a new highest value"""

    def __init__(
        self,
        obj: Callable,
        max_no_increasing_calls: int,
        key: Optional[Any] = None,
        name: str = None,
    ) -> None:
        super().__init__(obj, key, _logger, name)
        self._highest = -inf
        self._max_no_increase = max_no_increasing_calls
        self._calls = 0

    def __call__(self) -> bool:
        self._calls += 1
        if self._calls >= self._max_no_increase:
            if self._calls == self._max_no_increase:
                self.info(
                    "Highest value {} DID NOT increase after {} consecutive calls",
                    self._highest,
                    self._max_no_increase,
                )
            return True
        value = self._process_value()
        if value is not None and value < self._highest:
            self.info("New highest value {} (previous was {})", value, self._highest)
            self._highest = value
            self._calls = 0
        return False

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["highest"] = self._highest
        state["calls"] = self._calls
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._highest = state["highest"]
        self._calls = state["calls"]
