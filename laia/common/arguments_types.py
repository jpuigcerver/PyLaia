import logging
from argparse import ArgumentTypeError
from ast import literal_eval
from collections import OrderedDict


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected")


def str2num_accept_closed_range(v, t, vmin=None, vmax=None):
    v = t(v)
    if vmin is not None and v < vmin:
        raise ArgumentTypeError(
            "Value {} must be greater " "or equal than {}".format(v, vmin)
        )
    if vmax is not None and v > vmax:
        raise ArgumentTypeError(
            "Value {} must be lower " "or equal than {}".format(v, vmax)
        )
    return v


def str2num_accept_open_range(v, t, vmin=None, vmax=None):
    v = t(v)
    if vmin is not None and v <= vmin:
        raise ArgumentTypeError("Value {} must be greater than {}".format(v, vmin))
    if vmax is not None and v >= vmax:
        raise ArgumentTypeError("Value {} must be lower than {}".format(v, vmax))
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
        raise ArgumentTypeError("Valid logging levels are: {!r}", levels.values())


class NumberInClosedRange:
    def __init__(self, type, vmin=None, vmax=None):
        self._type = type
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, v):
        return str2num_accept_closed_range(v, self._type, self._vmin, self._vmax)


class NumberInOpenRange:
    def __init__(self, type, vmin=None, vmax=None):
        self._type = type
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, v):
        return str2num_accept_open_range(v, self._type, self._vmin, self._vmax)


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
                raise ArgumentTypeError("An element of {} is not a {}", x, self._type)
            if len(x) != self._dimensions:
                raise ArgumentTypeError("The given tuple does not match the dimensions")
            return tuple(x)
        else:
            raise ArgumentTypeError("{!r} is neither a tuple nor {}", v, self._type)
