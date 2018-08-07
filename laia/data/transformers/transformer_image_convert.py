from __future__ import absolute_import

from PIL import Image
from laia.data.transformers.transformer import Transformer


class TransformerImageConvert(Transformer):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode="L"):
        # type: (float) -> None
        super(TransformerImageConvert, self).__init__()
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, x):
        # type: (Image) -> Image
        return x.convert(self.mode)
