from __future__ import absolute_import

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from laia.data import PaddedTensor


def image_to_sequence(x, columnwise=True, return_packed=False):
    x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)

    if x.dim() == 2:
        x = x.view(1, 1, x.size()[0], x.size()[1])
    elif x.dim() == 3:
        x = x.view(1, x.size()[0], x.size()[1], x.size()[2])
    assert x.dim() == 4

    n, c, h, w = x.size()
    if columnwise:
        x = x.permute(3, 0, 1, 2).contiguous().view(w, n, h * c)
    else:
        x = x.permute(2, 0, 1, 3).contiguous().view(h, n, w * c)

    if xs is None:
        return x
    else:
        if columnwise:
            xs = xs.data[:, 1] if isinstance(xs, Variable) else xs[:, 1]
        else:
            xs = xs.data[:, 0] if isinstance(xs, Variable) else xs[:, 0]

        if return_packed:
            return pack_padded_sequence(x, xs.tolist())
        else:
            return x, xs.tolist()


class ImageToSequence(torch.nn.Module):
    def __init__(self, columnwise=True, return_packed=False):
        super(ImageToSequence, self).__init__()
        self._columnwise = columnwise
        self._return_packed = return_packed

    def forward(self, x):
        return image_to_sequence(
            x, columnwise=self._columnwise, return_packed=self._return_packed
        )
