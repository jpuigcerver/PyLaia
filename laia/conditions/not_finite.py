from typing import Callable, Any, Optional

import numpy as np

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class NotFinite(LoggingCondition):
    def __init__(
        self, obj: Callable, key: Optional[Any] = None, name: str = None
    ) -> None:
        super().__init__(obj, key, _logger, name)

    def __call__(self) -> bool:
        value = self._process_value()
        if value is None:
            return False
        if not np.isfinite(value):
            self.info("Value read from meter ({}) is not finite!", value)
            return True
        return False
