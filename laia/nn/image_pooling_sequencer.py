import re

import torch

from laia.data import PaddedTensor
from laia.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from laia.nn.image_to_sequence import image_to_sequence


class ImagePoolingSequencer(torch.nn.Module):
    def __init__(self, sequencer, columnwise=True):
        super().__init__()

        m = re.match(r"^(avgpool|maxpool|none)-([1-9][0-9]*)$", sequencer)
        if m is None:
            raise ValueError("The value of the sequencer argument is not valid")

        self._columnwise = columnwise
        self._fix_size = int(m.group(2))
        if m.group(1) == "avgpool":
            self.sequencer = AdaptiveAvgPool2d(
                (self._fix_size, None) if columnwise else (None, self._fix_size)
            )
        elif m.group(1) == "maxpool":
            self.sequencer = AdaptiveMaxPool2d(
                (self._fix_size, None) if columnwise else (None, self._fix_size)
            )

        else:
            # Assume that the images have a fixed height
            # (or width if columnwise=False)
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
                    and torch.sum(xs[:, 0] == self._fix_size).item() != ns
                ):
                    raise ValueError(
                        "Input images must have a fixed height "
                        "of {} pixels".format(self._fix_size)
                    )
                elif (
                    not self._columnwise
                    and torch.sum(xs[:, 1] == self._fix_size).item() != ns
                ):
                    raise ValueError(
                        "Input images must have a fixed width "
                        "of {} pixels".format(self._fix_size)
                    )
            else:
                if self._columnwise and x.size(-2) != self._fix_size:
                    raise ValueError(
                        "Input images must have a fixed height of {} pixels, "
                        "size is {}".format(self._fix_size, str(x.size()))
                    )
                elif not self._columnwise and x.size(-1) != self._fix_size:
                    raise ValueError(
                        "Input images must have a fixed width of {} pixels, "
                        "size is {}".format(self._fix_size, str(x.size()))
                    )

        x = image_to_sequence(x, columnwise=self._columnwise, return_packed=True)
        return x
