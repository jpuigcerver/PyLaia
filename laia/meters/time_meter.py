import time

from .meter import Meter

class TimeMeter(Meter):
    def reset(self):
        self._time = time.time()

    @property
    def value(self):
        return time.time() - self._time
