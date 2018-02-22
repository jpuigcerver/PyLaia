from __future__ import absolute_import

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from laia.data import PaddedTensor


class ImageColumnsToSequence(torch.nn.Module):
    def __init__(self, rows, return_packed=False):
        super(ImageColumnsToSequence, self).__init__()
        self._rows = rows
        self._return_packed = return_packed

    def forward(self, x):
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        if x.dim() == 2:
            x = x.view(1, 1, x.size()[0], x.size()[1])
        elif x.dim() == 3:
            x = x.view(1, x.size()[0], x.size()[1], x.size()[2])
        assert x.dim() == 4
        assert x.size()[2] == self._rows
        n, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(w, n, h * c)
        if xs is None:
            return x
        else:
            if self._return_packed:
                return pack_padded_sequence(x, list(xs[:, 1].data))
            else:
                return x, list(xs[:, 1].data)
