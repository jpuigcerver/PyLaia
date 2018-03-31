from __future__ import absolute_import

from laia.hooks.conditions.condition import Condition


class Always(Condition):
    def __init__(self):
        super(Always, self).__init__(obj=lambda: True)

    def __call__(self):
        return self._process_value()
