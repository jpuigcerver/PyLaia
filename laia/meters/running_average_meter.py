import math
import numpy as np
import torch

from .meter import Meter

class RunningAverageMeter(Meter):
    def reset(self):
        self._n  = 0.0
        self._s  = 0.0
        self._s2 = 0.0

    def add(self, v):
        if torch.is_tensor(v):
            self._n  += v.numel()
            self._s  += torch.sum(v)
            self._s2 += torch.sum(v * v)
        elif isinstance(v, torch.autograd.Variable) and torch.is_tensor(v.data):
            self._n  += v.data.numel()
            self._s  += torch.sum(v.data)
            self._s2 += torch.sum(v.data * v.data)
        elif isinstance(v, np.ndarray):
            self._n  += v.size
            self._s  += np.sum(v)
            self._s2 += np.sum(v * v)
        elif isinstance(v, (list, tuple)):
            self._n  += len(v)
            self._s  += sum(v)
            self._s2 += sum(map(lambda x: x * x, v))
        else:
            self._n  += 1
            self._s  += v
            self._s2 += v * v

    @property
    def value(self):
        avg = float(self._s) / float(self._n)
        var = float(self._s2) / float(self._n) - avg * avg
        return avg, math.sqrt(var)
