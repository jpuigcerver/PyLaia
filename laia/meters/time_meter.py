from __future__ import absolute_import

from laia.meters.meter import Meter

import time


class TimeMeter(Meter):
    def reset(self):
        self._time = time.time()

    @property
    def value(self):
        return time.time() - self._time
