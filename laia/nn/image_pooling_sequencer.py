from __future__ import absolute_import

import re

import torch
from laia.nn.adaptive_avgpool_2d import AdaptiveAvgPool2d
from laia.nn.adaptive_maxpool_2d import AdaptiveMaxPool2d
from laia.nn.image_to_sequence import image_to_sequence


class ImagePoolingSequencer(torch.nn.Module):
    def __init__(self, sequencer, columnwise=True):
        super(ImagePoolingSequencer, self).__init__()

        m = re.match(r'^(avgpool|maxpool|none)-([1-9][0-9]*)$', sequencer)
        if m is None:
            raise ValueError('The value of the sequencer argument is not valid')

        self._columnwise = columnwise
        self._fix_size = int(m.group(2))
        if m.group(1) == 'avgpool':
            self.sequencer = AdaptiveAvgPool2d((self._fix_size, None)
                                               if columnwise
                                               else (None, self._fix_size))
        elif m.group(1) == 'maxpool':
            self.sequencer = AdaptiveMaxPool2d((self._fix_size, None)
                                               if columnwise
                                               else (None, self._fix_size))
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
        x = image_to_sequence(x, columnwise=self._columnwise,
                              return_packed=True)
        return x
