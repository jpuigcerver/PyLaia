from __future__ import absolute_import

import numpy as np
from PIL import Image, ImageOps

from laia.data.transformers.transformer import Transformer


class TransformerImageInvert(Transformer):
    """Invert the colors of a PIL image with the given probability."""

    def __init__(self):
        # type: (float) -> None
        super(TransformerImageInvert, self).__init__()

    def __call__(self, x):
        # type: (Image) -> Image
        return ImageOps.invert(x)
