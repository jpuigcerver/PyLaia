from __future__ import absolute_import

from typing import Callable, Any

import laia.logging as log
from laia.hooks.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class TargetReached(LoggingCondition):
    """True if the target is reached"""

    def __init__(self, obj, target, key=None, name=None):
        # type: (Callable, int, Any, str) -> None
        super(TargetReached, self).__init__(obj, key, _logger, name)
        self._target = target

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        if self._target <= value:
            self.info('The target {} has been reached with value {}',
                      self._target, value)
            return True
        return False
