import logging
from argparse import ArgumentTypeError
from ast import literal_eval
from collections import OrderedDict


def str2num(v, t, vmin=None, vmax=None, open=False):
    v = t(v)
    if vmin is not None and (v <= vmin if open else v < vmin):
        raise ArgumentTypeError(f"Value {v} must be greater or equal than {vmin}")
    if vmax is not None and (v >= vmax if open else v > vmax):
        raise ArgumentTypeError(f"Value {v} must be lower or equal than {vmax}")
    return v


def str2loglevel(v):
    levels = OrderedDict(
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ]
    )
    try:
        return levels[v.lower()]
    except KeyError:
        raise ArgumentTypeError(f"Valid logging levels are: {levels.values()}")


class NumberInClosedRange:
    def __init__(self, type, vmin=None, vmax=None):
        self._type = type
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, v):
        return str2num(v, self._type, self._vmin, self._vmax, open=False)


class NumberInOpenRange:
    def __init__(self, type, vmin=None, vmax=None):
        self._type = type
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, v):
        return str2num(v, self._type, self._vmin, self._vmax, open=True)


class TupleList:
    def __init__(self, type, dimensions=2):
        assert dimensions >= 2
        self._type = type
        self._dimensions = dimensions

    def __call__(self, v):
        x = literal_eval(v)
        if isinstance(x, self._type):
            return (x,) * self._dimensions
        elif isinstance(x, (tuple, list)):
            if not all(type(v) == self._type for v in x):
                raise ArgumentTypeError(f"An element of {x} is not a {self._type}")
            if len(x) != self._dimensions:
                raise ArgumentTypeError("The given tuple does not match the dimensions")
            return tuple(x)
        else:
            raise ArgumentTypeError(f"{v} is neither a tuple nor {self._type}")
