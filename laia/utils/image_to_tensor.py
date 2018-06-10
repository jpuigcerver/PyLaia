from __future__ import division
import numpy as np
import torch

from PIL import Image, ImageOps


class ImageToTensor(object):
    def __init__(
        self,
        invert=True,
        mode="L",
        fixed_height=None,
        fixed_width=None,
        min_height=None,
        min_width=None,
        pad_color=0,
    ):
        assert mode in ("L", "RGB", "RGBA")
        assert fixed_height is None or fixed_height > 0
        assert fixed_width is None or fixed_width > 0
        assert min_height is None or min_height > 0
        assert min_width is None or min_width > 0
        self._invert = invert
        self._mode = mode
        self._fh = fixed_height
        self._fw = fixed_width
        self._mh = min_height
        self._mw = min_width
        self._pad_color = pad_color

    def __call__(self, x):
        assert isinstance(x, Image.Image)
        x = x.convert(self._mode)
        # Invert colors of the image
        if self._invert:
            x = ImageOps.invert(x)
        if self._fh or self._fw:
            # Optionally, Resize image to a fixed size
            cw, ch = x.size
            nw = self._fw if self._fw else int(cw * self._fh / ch)
            nh = self._fh if self._fh else int(ch * self._fw / cw)
            x = x.resize((nw, nh), Image.BILINEAR)
        elif self._mh or self._mw:
            # Optionally, pad image to have the minimum size
            cw, ch = x.size
            nw = cw if self._mw is None or cw >= self._mw else self._mw
            nh = ch if self._mh is None or ch >= self._mh else self._mh
            if cw != nw or ch != nh:
                nx = Image.new("L", size=(nw, nh), color=self._pad_color)
                nx.paste(x, ((nw - cw) // 2, (nh - ch) // 2))
                x = nx

        x = np.asarray(x, dtype=np.float32)
        if len(x.shape) != 3:
            x = np.expand_dims(x, axis=-1)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x / 255.0)
