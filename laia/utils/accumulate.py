from __future__ import absolute_import

import operator

try:
    from itertools import accumulate
except ImportError:
    def accumulate(iterable, func=operator.add):
        r"""Make an iterator that returns accumulated sums.

        Accumulated results for other binary functions can be implemented
        if a function of two arguments is given through the ``func`` argument.
        Elements of the input iterable may be any type that can be accepted as
        arguments to func.

        Backport implementation to Python2 of itertools.accumulate
        """

        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return

        yield total
        for element in it:
            total = func(total, element)
            yield total
