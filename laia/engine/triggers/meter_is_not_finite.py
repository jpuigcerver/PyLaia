from __future__ import absolute_import

import numpy as np
from laia.engine.triggers.trigger import TriggerLogWrapper
from laia.engine.triggers.trigger_from_meter import TriggerFromMeter


class MeterIsNotFinite(TriggerFromMeter):
    def __init__(self, meter, meter_key=None, name=None):
        super(MeterIsNotFinite, self).__init__(meter, meter_key, name)

    def _process_value(self, last_value):
        if not np.isfinite(last_value):
            self.logger.info(TriggerLogWrapper(
                self, 'Value read from meter ({}) is not finite!',
                last_value))
            return True
        else:
            return False
