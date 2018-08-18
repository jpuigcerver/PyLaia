from __future__ import absolute_import

import numpy as np
from typing import Callable, Union, Tuple


class Transformer(object):
    def __call__(self, x):
        return x

    def __str__(self):
        return self._to_string(0)

    def _type(self):
        return self.__class__.__name__

    def _to_string(self, spaces):
        return (" " * spaces) + self._type()


class TransformerList(Transformer):
    """Apply a sequence of transformations to the input sample."""

    def __init__(self, *transformers):
        # type: (*Callable) -> None
        super(TransformerList, self).__init__()
        assert transformers, "You must specify at least one choice"
        self._transformers = transformers

    def __call__(self, x):
        for transformer in self._transformers:
            x = transformer(x)
        return x

    def _to_string(self, spaces):
        s = (" " * spaces) + "{}[\n".format(self._type())
        for transformer in self._transformers:
            if hasattr(transformer, "_to_string"):
                s += transformer._to_string(spaces + 2) + ",\n"
            elif hasattr(transformer, "__str__"):
                s += (" " * (spaces + 2)) + str(transformer) + ",\n"
            else:
                s += (" " * (spaces + 2)) + repr(transformer) + ",\n"
        s += (" " * spaces) + "]"
        return s


class TransformerConditional(Transformer):
    """Apply a transformation to the input sample with some probability."""

    def __init__(self, transformer, p=0.5):
        # type: (Callable, float) -> None
        super(TransformerConditional, self).__init__()
        self._transformer = transformer
        self._p = p

    def __call__(self, x):
        if np.random.rand() < self._p:
            return self._transformer(x)
        else:
            return x

    def _to_string(self, spaces):
        s = (" " * spaces) + "%s(probability=%g)[\n" % (self._type(), self._p)
        if hasattr(self._transformer, "_to_string"):
            s += self._transformer._to_string(spaces + 2) + "\n"
        elif hasattr(self._transformer, "__str__"):
            s += (" " * (spaces + 2)) + str(self._transformer) + "\n"
        else:
            s += (" " * (spaces + 2)) + repr(self._transformer) + "\n"
        s += (" " * spaces) + "]"
        return s


class TransformerChoice(Transformer):
    """Apply a randomly transformation chosen from a given set with some probability."""

    def __init__(self, *transformers):
        # type: (*Union[Callable, Tuple[float, Callable]]) -> None
        super(TransformerChoice, self).__init__()
        assert transformers, "You must specify at least one choice"

        self._transformers = []
        self._probs = []
        for transformer in transformers:
            if isinstance(transformer, tuple):
                self._probs.append(transformer[0])
                self._transformers.append(transformer[1])
            else:
                self._transformers.append(transformer)
        if self._probs:
            assert len(self._probs) == len(self._transformers)
        else:
            self._probs = None

    def __call__(self, x):
        t = np.random.choice(np.arange(len(self._transformers)), p=self._probs)
        return self._transformers[t](x)

    def _to_string(self, spaces):
        s = (" " * spaces) + "{}[\n".format(self._type())
        for transformer in self._transformers:
            if hasattr(transformer, "_to_string"):
                s += transformer._to_string(spaces + 2) + ",\n"
            elif hasattr(transformer, "__str__"):
                s += (" " * (spaces + 2)) + str(transformer) + ",\n"
            else:
                s += (" " * (spaces + 2)) + repr(transformer) + ",\n"
        s += (" " * spaces) + "]"
        return s
