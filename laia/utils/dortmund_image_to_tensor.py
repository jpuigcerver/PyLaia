from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


def dortmund_distort(img, random_limits=(0.8, 1.1)):
    """
    Creates an augmentation by computing a homography from three points in the
    image to three randomly generated points.
    """
    y, x = img.shape[:2]
    src_point = np.float32([[x / 2, y / 3],
                            [2 * x / 3, 2 * y / 3],
                            [x / 3, 2 * y / 3]])
    random_shift = (np.random.rand(3, 2) - 0.5) * 2 * (
            random_limits[1] - random_limits[0]) / 2 + np.mean(
        random_limits)
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if img.ndim == 3:
        border_value = np.median(
            np.reshape(img, (img.shape[0] * img.shape[1], -1)), axis=0)
    else:
        border_value = float(np.median(img))
    return cv2.warpAffine(img, transform, dsize=(x, y),
                          borderValue=border_value)


class DortmundImageToTensor(object):
    def __init__(self, fixed_height=None, fixed_width=None):
        assert fixed_height is None or fixed_height > 0
        assert fixed_width is None or fixed_width > 0
        self._fh = fixed_height
        self._fw = fixed_width

    def __call__(self, x):
        assert isinstance(x, Image.Image)
        x = x.convert('L')
        x = ImageOps.invert(x)
        x = np.asarray(x, dtype=np.float32)
        x = dortmund_distort(x / 255.0)
        # Optionally, resize image to a fixed size
        if self._fh or self._fw:
            ch, cw = x.shape[:2]
            nw = self._fw if self._fw else int(cw * self._fh / ch)
            nh = self._fh if self._fh else int(ch * self._fw / cw)
            x = cv2.resize(x, dsize=(nw, nh), interpolation=cv2.INTER_LINEAR)

        if x.shape != 3:
            x = np.expand_dims(x, axis=-1)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)
