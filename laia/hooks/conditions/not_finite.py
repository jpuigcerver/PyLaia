from __future__ import absolute_import

import numpy as np

from laia.engine.conditions.condition import ConditionFromMeter


class NotFinite(ConditionFromMeter):
    def __init__(self, meter, meter_key=None, name=None):
        super(NotFinite, self).__init__(meter, meter_key, name)

    def _process_value(self, last_value):
        if not np.isfinite(last_value):
            self.logger.info(
                'Value read from meter ({}) is not finite!',
                last_value)
            return True
        else:
            return False
