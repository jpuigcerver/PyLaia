from typing import Callable, Any

import laia.common.logging as log
from laia.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class GEqThan(LoggingCondition):
    """True if greater or equal than a target"""

    def __init__(self, obj, target, key=None, name=None):
        # type: (Callable, Any, Any, str) -> None
        super(GEqThan, self).__init__(obj, key, _logger, name)
        self._target = target

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        if value >= self._target:
            self.info(
                "The target {} has been reached with value {}", self._target, value
            )
            return True
        return False
