import logging
from argparse import ArgumentTypeError
from ast import literal_eval
from typing import Optional, Type, Union


def str2num(
    v: str,
    t: Type[Union[int, float]],
    vmin: Optional[Union[int, float]] = None,
    vmax: Optional[Union[int, float]] = None,
    open: bool = False,
) -> Union[int, float]:
    v = t(v)
    vmin = float("-inf") if vmin is None else t(vmin)
    vmax = float("inf") if vmax is None else t(vmax)
    assert vmin <= vmax
    if (open and not vmin < v < vmax) or (not open and not vmin <= v <= vmax):
        raise ArgumentTypeError(
            f"Value {v} must be in the "
            f"{'(' if open else '['}"
            f"{vmin}, {vmax}"
            f"{')' if open else ']'} interval"
        )
    return v


def str2loglevel(v: str) -> int:
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    v = v.lower()
    try:
        return levels[v]
    except KeyError:
        raise ArgumentTypeError(
            f'Valid logging levels are {list(levels.keys())}, given "{v}"'
        )


class NumberInClosedRange:
    def __init__(
        self,
        type: Type[Union[int, float]],
        vmin: Optional[Union[int, float]] = None,
        vmax: Optional[Union[int, float]] = None,
    ):
        self.type = type
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, v: str):
        return str2num(v, self.type, vmin=self.vmin, vmax=self.vmax, open=False)


class NumberInOpenRange:
    def __init__(
        self,
        type: Type[Union[int, float]],
        vmin: Optional[Union[int, float]] = None,
        vmax: Optional[Union[int, float]] = None,
    ):
        self.type = type
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, v: str):
        return str2num(v, self.type, vmin=self.vmin, vmax=self.vmax, open=True)


class TupleList:
    def __init__(self, type: Type, dimensions: int = 2):
        assert dimensions > 1
        self.type = type
        self.dimensions = dimensions

    def __call__(self, v):
        x = literal_eval(v)
        if isinstance(x, self.type):
            return (x,) * self.dimensions
        elif isinstance(x, (tuple, list)):
            if not all(type(v) == self.type for v in x):
                raise ArgumentTypeError(f"An element of {x} is not a {self.type}")
            if len(x) != self.dimensions:
                raise ArgumentTypeError(
                    f"The given input {x} does not match "
                    f"the given dimensions {self.dimensions}"
                )
            return tuple(x)
        else:
            raise ArgumentTypeError(f"{v} is neither a tuple nor {self.type}")
