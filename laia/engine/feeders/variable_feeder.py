from __future__ import absolute_import

from laia.data import PaddedTensor
from laia.engine.feeders.feeder import Feeder

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence


class VariableFeeder(Feeder):
    r"""Feed a PyTorch Variable to the model.

    Args:
      device (int): integer representing the device where the data should
        be allocated. Important: gpu devices start from 1.
      requires_grad (boolean, optional): whether or not the `Variable`
          requires grads. (default: False)
      parent_feeder (callable, optional): parent feeder that should feed this.
          (default: None)
    """
    def __init__(self, device, requires_grad=False, parent_feeder=None):
        super(VariableFeeder, self).__init__(parent_feeder)
        self._device = device
        self._requires_grad = requires_grad

    def _feed(self, batch):
        if torch.is_tensor(batch):
            if self._device > 0:
                batch = batch.cuda(self._device - 1)
            else:
                batch = batch.cpu()
            return Variable(data=batch, requires_grad=self._requires_grad)

        elif isinstance(batch, PaddedTensor):
            x, xs = batch.data, batch.sizes
            x = x.data if isinstance(x, Variable) else x
            xs = xs.data if isinstance(xs, Variable) else xs

            if self._device > 0:
                x, xs = x.cuda(self._device - 1), xs.cuda(self._device - 1)
            else:
                x, xs = x.cpu(), xs.cpu()

            x = Variable(data=x, requires_grad=self._requires_grad)
            xs = Variable(data=xs, requires_grad=False)
            return PaddedTensor(data=x, sizes=xs)

        elif isinstance(batch, PackedSequence):
            x, xs = batch.data, batch.batch_sizes
            x = x.data if isinstance(x, Variable) else x
            xs = xs.data if isinstance(xs, Variable) else xs

            if self._device > 0:
                x = x.cuda(self._device - 1)
                if torch.is_tensor(xs):
                    xs = xs.cuda(self._device - 1)
            else:
                x = x.cpu()
                if torch.is_tensor(xs):
                    xs = xs.cpu()

            x = Variable(data=x, requires_grad=self._requires_grad)
            if torch.is_tensor(xs):
                xs = Variable(data=xs, requires_grad=False)
            return PackedSequence(data=x, batch_sizes=xs)

        else:
            raise ValueError('Type {!r} is not supported'.format(type(batch)))
