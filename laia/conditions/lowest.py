from typing import Callable, Any, Optional, Dict

from torch._six import inf

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class Lowest(LoggingCondition):
    """True if a new lowest value has been reached"""

    def __init__(
        self, obj: Callable, key: Optional[Any] = None, name: str = None
    ) -> None:
        super().__init__(obj, key, _logger, name)
        self._lowest = inf

    def __call__(self) -> bool:
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

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["lowest"] = self._lowest
        return state

    def load_state_dict(self, state: Dict) -> None:
        super().load_state_dict(state)
        self._lowest = state["lowest"]
