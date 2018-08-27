from typing import Callable, Any, Optional

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class MultipleOf(LoggingCondition):
    """True if the dividend is a multiple of the divisor"""

    def __init__(
        self, obj: Callable, divisor: int, key: Optional[Any] = None, name: str = None
    ) -> None:
        assert divisor > 0
        super().__init__(obj, key, _logger, name)
        self._divisor = divisor

    def __call__(self) -> bool:
        value = self._process_value()
        if value is None:
            return False
        if value % self._divisor == 0:
            self.info("{} is a multiple of {}", value, self._divisor)
            return True
        return False
