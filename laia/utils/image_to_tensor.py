import numpy as np
import torch

from PIL import Image, ImageOps


class ImageToTensor(object):

    def __init__(self, invert=True, mode="L", fixed_height=None, fixed_width=None):
        assert mode in ("L", "RGB", "RGBA")
        assert fixed_height is None or fixed_height > 0
        assert fixed_width is None or fixed_width > 0
        self._invert = invert
        self._mode = mode
        self._fh = fixed_height
        self._fw = fixed_width

    def __call__(self, x):
        assert isinstance(x, Image.Image)
        x = x.convert(self._mode)
        # Resize image to a fixed size
        if self._fh or self._fw:
            cw, ch = x.size
            nw = self._fw if self._fw else int(cw * self._fh / ch)
            nh = self._fh if self._fh else int(ch * self._fw / cw)
            x = x.resize((nw, nh), Image.BILINEAR)
        # Invert colors of the image
        if self._invert:
            x = ImageOps.invert(x)

        x = np.asarray(x, dtype=np.float32)
        if len(x.shape) != 3:
            x = np.expand_dims(x, axis=-1)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x / 255.0)
