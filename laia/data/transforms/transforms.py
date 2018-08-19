from __future__ import absolute_import

from typing import Callable, Union, Tuple, Sequence

import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms.transforms import RandomTransforms


class RandomProbChoice(RandomTransforms):
    """Apply a randomly transformation chosen from a given set with some probability."""

    def __init__(self, transforms):
        # type: (Sequence[Union[Callable, Tuple[float, Callable]]]) -> None
        super(RandomProbChoice, self).__init__(transforms)
        assert transforms, "You must specify at least one choice"

        self._transforms = []
        self._probs = []
        for transformer in transforms:
            if isinstance(transformer, tuple):
                self._probs.append(transformer[0])
                self._transforms.append(transformer[1])
            else:
                self._transforms.append(transformer)
        if self._probs:
            assert len(self._probs) == len(self._transforms)
        else:
            self._probs = None

    def __call__(self, x):
        t = np.random.choice(np.arange(len(self._transforms)), p=self._probs)
        return self._transforms[t](x)


class Invert(object):
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img):
        # type: (Image) -> Image
        return ImageOps.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Convert(object):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode):
        # type: (str) -> None
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img):
        # type: (Image) -> Image
        return img.convert(self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += "mode={}".format(self.mode)
        format_string += ")"
        return format_string
