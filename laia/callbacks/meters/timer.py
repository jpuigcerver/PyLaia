import time

from laia.callbacks.meters.meter import Meter


class Timer(Meter):
    def __init__(self):
        super().__init__()
        self.start = time.time()
        self.end = None

    def reset(self):
        self.start = time.time()
        self.end = None
        return self

    def stop(self):
        self.end = time.time()
        return self

    @property
    def value(self) -> float:
        if self.end is None:
            self.end = time.time()
        return self.end - self.start
