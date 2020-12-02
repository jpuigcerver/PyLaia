import time

from laia.callbacks.meters.meter import Meter


class Timer(Meter):
    def __init__(self):
        super().__init__()
        self.start = self.time()
        self.end = None

    @staticmethod
    def time() -> float:
        return time.time()

    def reset(self):
        self.start = self.time()
        self.end = None
        return self

    def stop(self):
        self.end = self.time()
        return self

    @property
    def value(self) -> float:
        if self.end is None:
            self.end = self.time()
        return self.end - self.start
