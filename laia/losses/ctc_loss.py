from __future__ import absolute_import

import itertools

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from warpctc_pytorch import CTCLoss as _CTCLoss

from laia.losses.loss import Loss


class CTCLoss(Loss):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self._ctc = _CTCLoss()

    def __call__(self, output, target):
        assert isinstance(output, PackedSequence)
        x, xs = pad_packed_sequence(output)
        with torch.cuda.device_of(x):
            xs = Variable(torch.IntTensor(xs))
            y = Variable(
                torch.IntTensor(list(itertools.chain.from_iterable(target))))
            ys = Variable(torch.IntTensor(map(lambda x: len(x), target)))
        self._loss = self._ctc(x, y, xs, ys)
        return (self._loss / xs.type(torch.FloatTensor)).mean()
