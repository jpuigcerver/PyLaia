from __future__ import absolute_import

import time

from laia.meters import Meter


class TimeMeter(Meter):
    def __init__(self, exceptions_threshold=5):
        super(TimeMeter, self).__init__(exceptions_threshold)
        self._start = time.time()
        self._end = None

    def reset(self):
        self._start = time.time()
        self._end = None
        return self

    def stop(self):
        self._end = time.time()
        return self

    @property
    def value(self):
        if self._end is None:
            self._end = time.time()
        return self._end - self._start
