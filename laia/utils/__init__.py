from __future__ import absolute_import

from laia.utils.checks import check_inf, check_nan
from laia.utils.image_collage import image_collage
from laia.utils.image_to_tensor import ImageToTensor
from laia.utils.symbols_table import SymbolsTable
from laia.utils.text_to_tensor import TextToTensor
from laia.utils.phoc import unigram_phoc, TextToPHOC

import laia.utils.logging


try:
    from itertools import accumulate
except ImportError:
    import operator

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
