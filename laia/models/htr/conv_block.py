from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

from laia.data import PaddedTensor


def _get_channels(x):
    return x.data.size(1) if isinstance(x, PaddedTensor) else x.size(1)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        activation=nn.LeakyReLU,
        poolsize=None,
        dropout=None,
        batchnorm=False,
        inplace=False,
    ):
        super(ConvBlock, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(dilation, (list, tuple)):
            dilation = (dilation, dilation)
        if poolsize and not isinstance(poolsize, (list, tuple)):
            poolsize = (poolsize, poolsize)

        self.in_channels = in_channels

        # This is a torch tensor used to divide the input sizes in the
        # forward() method, when using a PaddedTensor as an input.
        # If no padding is effectively used, then this is set to None.
        self.register_buffer(
            "poolsize",
            (
                torch.LongTensor(poolsize)
                if poolsize
                and poolsize[0] > 0
                and poolsize[1] > 0
                and (poolsize[0] > 1 or poolsize[1] > 1)
                else None
            ),
        )

        # First, add dropout module (optionally)
        if dropout:
            self.add_module("dropout", nn.Dropout(dropout, inplace=inplace))

        # Add Conv2d layer (compute padding to perform a full convolution).
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=tuple(
                    (kernel_size[dim] - 1) // 2 * dilation[dim] for dim in (0, 1)
                ),
                dilation=dilation,
            ),
        )

        # Add Batch normalization
        if batchnorm:
            self.add_module("batchnorm", nn.BatchNorm2d(out_channels))

        # Activation function must support inplace operations.
        self.add_module("activation", activation(inplace=inplace))

        # Add maxpool layer
        if self.poolsize is not None:
            self.add_module("pooling", nn.MaxPool2d(poolsize))

    def forward(self, x):
        assert _get_channels(x) == self.in_channels, (
            "Input image depth ({}) does not match the "
            "expected ({})".format(_get_channels(x), self.in_channels)
        )
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        # Forward the input through all the modules in the conv block
        for module in self._modules.values():
            x = module(x)
        if xs is None:
            return x
        else:
            assert xs.dim() == 2, "PaddedTensor.sizes must be a matrix"
            assert xs.size(1) == 2, (
                "PaddedTensor.sizes must have 2 columns: Height and Width, "
                "{} columns given instead.".format(xs.size(1))
            )
            ys = torch.zeros_like(xs.data)
            for dim in 0, 1:
                ys[:, dim] = self.get_output_size(xs.data, dim)
                if self.poolsize is not None:
                    ys[:, dim] /= self.poolsize[dim]
            return PaddedTensor(x, sizes=Variable(ys))

    def get_output_size(self, input_, dim):
        return (
            input_[:, dim]
            + 2 * self.conv.padding[dim]
            - self.conv.dilation[dim] * (self.conv.kernel_size[dim] - 1)
            - 1
        ) / self.conv.stride[dim] + 1
