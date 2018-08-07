from __future__ import absolute_import

import numpy as np
from PIL import Image, ImageOps
from laia.data.transformers.transformer import Transformer


class TransformerImageInvert(Transformer):
    """Invert the colors of a PIL image with the given probability."""

    def __init__(self, probability=0.5):
        # type: (float) -> None
        super(TransformerImageInvert, self).__init__()
        self.probability = probability

    def __call__(self, x):
        # type: (Image) -> Image
        if np.random.random() < self.probability:
            return ImageOps.invert(x)
        else:
            return x
