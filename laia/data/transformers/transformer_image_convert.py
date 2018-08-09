from __future__ import absolute_import

from PIL import Image
from typing import AnyStr

from laia.data.transformers.transformer import Transformer

import numpy as np


class TransformerImageConvert(Transformer):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode, probability=None):
        # type: (AnyStr, float) -> None
        super(TransformerImageConvert, self).__init__()
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode
        self.probability = probability

    def __call__(self, x):
        # type: (Image) -> Image
        if self.probability is None or np.random.rand() < self.probability:
            return x.convert(self.mode)
        else:
            return x
