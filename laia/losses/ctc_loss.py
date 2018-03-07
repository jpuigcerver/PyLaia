from __future__ import absolute_import

import itertools

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from warpctc_pytorch import CTCLoss as _CTCLoss

from laia.losses.loss import Loss


class CTCLoss(Loss):
    def __init__(self, size_average=True):
        super(CTCLoss, self).__init__()
        self._ctc = _CTCLoss(size_average)

    def __call__(self, output, target):
        assert isinstance(output, PackedSequence)
        x, xs = pad_packed_sequence(output)
        assert xs[0] == x.size(0), 'Maximum length does not match'
        assert len(target) == x.size(1), 'Batch size does not match'

        # Prepare tensors of the correct type
        y = torch.IntTensor(list(itertools.chain.from_iterable(target)))
        xs = torch.IntTensor(xs)
        ys = torch.IntTensor([len(y_) for y_ in target])

        y = Variable(y, requires_grad=False)
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)

        # Compute Loss
        self._loss = self._ctc(x, y, xs, ys)
        #self._loss = torch.sum(x)
        #self._loss = Variable(torch.zeros(1), requires_grad=True)
        return self._loss
