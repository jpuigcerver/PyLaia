from __future__ import absolute_import

import re

import torch

from laia.data import PaddedTensor
from laia.nn.image_to_sequence import image_to_sequence

try:
    from laia.nn.adaptive_avgpool_2d import AdaptiveAvgPool2d
    from laia.nn.adaptive_maxpool_2d import AdaptiveMaxPool2d

    nnutils_is_available = True
except ImportError:
    nnutils_is_available = False


class ImagePoolingSequencer(torch.nn.Module):
    def __init__(self, sequencer, columnwise=True):
        super(ImagePoolingSequencer, self).__init__()

        m = re.match(r"^(avgpool|maxpool|none)-([1-9][0-9]*)$", sequencer)
        if m is None:
            raise ValueError("The value of the sequencer argument is not valid")

        self._columnwise = columnwise
        self._fix_size = int(m.group(2))
        if m.group(1) == "avgpool":
            if not nnutils_is_available:
                raise ImportError("nnutils does not seem installed")
            self.sequencer = AdaptiveAvgPool2d(
                (self._fix_size, None) if columnwise else (None, self._fix_size)
            )
        elif m.group(1) == "maxpool":
            if not nnutils_is_available:
                raise ImportError("nnutils does not seem installed")
            self.sequencer = AdaptiveMaxPool2d(
                (self._fix_size, None) if columnwise else (None, self._fix_size)
            )

        else:
            # Assume that the images have a fixed height (or width,
            # if columnwise=False)
            self.sequencer = None

    @property
    def columnwise(self):
        return self._columnwise

    @property
    def fix_size(self):
        return self._fix_size

    def forward(self, x):
        if self.sequencer:
            x = self.sequencer(x)
        else:
            if isinstance(x, PaddedTensor):
                xs = x.sizes  # batch sizes matrix (N x 2)
                ns = xs.size(0)  # number of samples in the batch
                if (
                    self._columnwise
                    and torch.sum(xs[:, 0] == self._fix_size).data[0] != ns
                ):
                    raise ValueError(
                        "Input images must have a fixed height of {} pixels".format(
                            self._fix_size
                        )
                    )
                elif (
                    not self._columnwise
                    and torch.sum(xs[:, 1] == self._fix_size).data[0] != ns
                ):
                    raise ValueError(
                        "Input images must have a fixed width of {} pixels".format(
                            self._fix_size
                        )
                    )
            else:
                if self._columnwise and x.size(-2) != self._fix_size:
                    raise ValueError(
                        "Input images must have a fixed height of {} pixels, "
                        "size is {}".format(self._fix_size, str(x.size()))
                    )
                elif (not self._columnwise) and x.size(-1) != self._fix_size:
                    raise ValueError(
                        "Input images must have a fixed width of {} pixels, "
                        "size is {}".format(self._fix_size, str(x.size()))
                    )

        x = image_to_sequence(x, columnwise=self._columnwise, return_packed=True)
        return x
