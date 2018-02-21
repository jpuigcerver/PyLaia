import numpy as np
import torch

from PIL import ImageOps


class ImageToTensor(object):
    def __init__(self, invert=True, mode='L'):
        assert mode in ('L', 'RGB', 'RGBA')
        self._invert = invert
        self._mode = mode

    def __call__(self, x):
        x = x.convert(self._mode)
        if self._invert:
            x = ImageOps.invert(x)
        x = np.asarray(x, dtype=np.float32)
        if len(x.shape) != 3:
            x = np.expand_dims(x, axis=-1)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x / 255.0)
