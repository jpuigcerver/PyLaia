from __future__ import absolute_import

import torch
from torch.autograd import Variable

from laia.data import PaddedTensor
from laia.engine.feeders.variable_feeder import VariableFeeder


class ImageFeeder(VariableFeeder):
    r"""Feed an image as a PyTorch Variable to the model.

    Args:
      device (int): integer representing the device where the data should
          be allocated. Important: gpu devices start from 1.
      keep_padded_tensors (boolean, optional): whether or not keep the size
          information of the padding. If False, the batch tensor will be
          returned without any size information. (default: True)
      keep_channels_in_size (boolean, optional): whether or not the number of
          channels of the images is kept as part of the size in the
          `PaddedTensor` objects.
      requires_grad (boolean, optional): whether or not the `Variable`
          requires grads. (default: False)
      parent_feeder (callable, optional): parent feeder that should feed this.
          (default: None)
    """

    def __init__(self, device, keep_padded_tensors=True,
                 keep_channels_in_size=False, requires_grad=False,
                 parent_feeder=None):
        super(ImageFeeder, self).__init__(device=device,
                                          requires_grad=requires_grad,
                                          parent_feeder=parent_feeder)
        self._keep_padded_tensors = keep_padded_tensors
        self._keep_channels_in_size = keep_channels_in_size

    @classmethod
    def _view_as_4d(cls, batch):
        if batch.dim() == 2:
            batch = batch.view(1, 1, batch.size(0), batch.size(1))
        elif batch.dim() == 3:
            batch = batch.view(1, batch.size(0), batch.size(1), batch.size(2))
        elif batch.dim() == 4:
            pass
        else:
            raise ValueError('Tensor with {} dimensions is not supported '
                             'as an Image'.format(batch.dim()))
        return batch

    def _feed(self, batch):
        batch = super(ImageFeeder, self)._feed(batch)

        if torch.is_tensor(batch.data):
            # View image batch as a N-C-H-W
            batch = self._view_as_4d(batch.data)
            return Variable(batch, requires_grad=self._requires_grad)

        elif isinstance(batch, PaddedTensor):
            # View image batch as a N-C-H-W
            x = self._view_as_4d(batch.data.data)
            if self._keep_padded_tensors:
                xs = batch.sizes.data
                # Ensure that the size tensor is the expected
                if xs.dim() != 2 or (xs.size(1) != 2 and xs.size(1) != 3):
                    raise ValueError('Size tensor in PaddedTensor has not an '
                                     'expected shape: {!r}'.format(xs.size()))

                if xs.size(1) == 3 and not self._keep_channels_in_size:
                    xs = xs[:, 1:]

                x = Variable(x, requires_grad=self._requires_grad)
                xs = Variable(xs)
                return PaddedTensor(x, sizes=xs)
            else:
                return Variable(x, requires_grad=self._requires_grad)
        else:
            raise ValueError('Type {!r} is not supported'.format(type(batch)))
