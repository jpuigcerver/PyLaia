from __future__ import absolute_import

from typing import Callable, Any

import numpy as np

import laia.logging as log
from laia.hooks.conditions.condition import LoggingCondition

_logger = log.get_logger(__name__)


class NotFinite(LoggingCondition):
    def __init__(self, obj, key=None, name=None):
        # type: (Callable, Any,str) -> None
        super(NotFinite, self).__init__(obj, key, _logger, name)

    def __call__(self):
        value = self._process_value()
        if value is None:
            return False
        if not np.isfinite(value):
            self.info('Value read from meter ({}) is not finite!', value)
            return True
        return False
