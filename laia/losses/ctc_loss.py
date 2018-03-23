from __future__ import absolute_import

import itertools

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from warpctc_pytorch import CTCLoss as _CTCLoss

from laia.losses.loss import Loss


class CTCLoss(Loss):
    def __init__(self, size_average=False, length_average=True):
        super(CTCLoss, self).__init__()
        self._ctc = _CTCLoss(size_average, length_average)

    def __call__(self, output, target):
        if isinstance(output, PackedSequence):
            x, xs = pad_packed_sequence(output)
        else:
            x, xs = output, [output.size(0)]
        assert xs[0] == x.size(0), 'Maximum length does not match'
        assert len(target) == x.size(1), 'Batch size does not match'

        # Prepare tensors of the correct type
        y = torch.IntTensor(list(itertools.chain.from_iterable(target)))
        xs = torch.IntTensor(xs)
        ys = torch.IntTensor([len(y_) for y_ in target])

        y = Variable(y)
        xs = Variable(xs)
        ys = Variable(ys)

        # Compute Loss
        self._loss = self._ctc(x, y, xs, ys)
        return self._loss
