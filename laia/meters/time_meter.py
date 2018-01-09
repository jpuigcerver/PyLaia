import time

from .meter import Meter

class TimeMeter(Meter):
    def __init__(self):
        super(TimeMeter, self).__init__()
        self._time = time.time()

    def reset(self):
        self._time = time.time()

    @property
    def value(self):
        return time.time() - self._time
