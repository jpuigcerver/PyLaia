from __future__ import division

import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, Union, Tuple

from laia.data import PaddedTensor

try:
    from laia.nn.mask_image_from_size import mask_image_from_size
except ImportError:
    import warnings

    mask_image_from_size = None


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # type: int
        out_channels,  # type: int
        kernel_size=3,  # type: Union[int, Tuple[int, int]]
        stride=1,  # type: Union[int, Tuple[int, int]]
        dilation=1,  # type: Union[int, Tuple[int, int]]
        activation=nn.LeakyReLU,  # type: Optional[nn.Module]
        poolsize=None,  # type: Optional[Union[int, Tuple[int, int]]]
        dropout=None,  # type: Optional[float]
        batchnorm=False,  # type: bool
        inplace=False,  # type: bool
        use_masks=False,  # type: bool
    ):
        # type: (...) -> None
        super(ConvBlock, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(dilation, (list, tuple)):
            dilation = (dilation,) * 2

        # Prepare poolsize
        if poolsize and not isinstance(poolsize, (list, tuple)):
            poolsize = (poolsize, poolsize)
        elif isinstance(poolsize, (list, tuple)):
            poolsize = (
                None
                if poolsize in product((0, 1), repeat=2)
                else tuple(poolsize[dim] if poolsize[dim] else 1 for dim in (0, 1))
            )

        if use_masks and mask_image_from_size is None:
            warnings.warn(
                "nnutils does not seem to be installed, masking cannot be used"
            )
            use_masks = False

        self.dropout = dropout
        self.in_channels = in_channels
        self.use_masks = use_masks
        self.poolsize = poolsize

        # Add Conv2d layer (compute padding to perform a full convolution).
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=tuple(
                (kernel_size[dim] - 1) // 2 * dilation[dim] for dim in (0, 1)
            ),
            dilation=dilation,
            # Note: If batchnorm is used, the bias does not affect the output
            # of the unit.
            bias=not batchnorm,
        )

        # Add Batch normalization
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # Activation function must support inplace operations.
        self.activation = activation(inplace=inplace) if activation else None

        # Add maxpool layer
        if self.poolsize:
            self.pool = nn.MaxPool2d(poolsize)
        else:
            self.pool = None

    def forward(self, x):
        # type: (Union[Variable, PaddedTensor]) -> Union[Variable, PaddedTensor]
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
            assert xs.dim() == 2, "PaddedTensor.sizes must be a matrix"
            assert xs.size(1) == 2, (
                "PaddedTensor.sizes must have 2 columns: Height and Width, "
                "{} columns given instead.".format(xs.size(1))
            )
            assert x.size(0) == xs.size(0), (
                "Number of batch sizes ({}) does not match the number of "
                "samples in the batch {}".format(xs.size(0), x.size(0))
            )
        else:
            xs = None
        assert x.size(1) == self.in_channels, (
            "Input image depth ({}) does not match the "
            "expected ({})".format(x.size(1), self.in_channels)
        )

        if self.dropout and 0.0 < self.dropout < 1.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv(x)
        if self.use_masks:
            x = mask_image_from_size(x, batch_sizes=xs, mask_value=0, inplace=True)

        if self.batchnorm:
            x = self.batchnorm(x)

        if self.activation:
            x = self.activation(x)

        if self.use_masks:
            x = mask_image_from_size(x, batch_sizes=xs, mask_value=0, inplace=True)

        if self.pool:
            x = self.pool(x)

        return (
            x if xs is None else PaddedTensor(x, sizes=self.get_output_batch_size(xs))
        )

    def get_output_batch_size(self, xs):
        if isinstance(xs, Variable):
            xs = xs.data
            return_variable = True
        else:
            return_variable = False
        ys = torch.zeros_like(xs)
        for dim in 0, 1:
            ys[:, dim] = self.get_output_size(
                size=xs[:, dim],
                kernel_size=self.conv.kernel_size[dim],
                dilation=self.conv.dilation[dim],
                stride=self.conv.stride[dim],
                poolsize=self.poolsize[dim] if self.poolsize else None,
                padding=self.conv.padding[dim],
            )
        return Variable(ys) if return_variable else ys

    @staticmethod
    def get_output_size(
        size,  # type: Union[torch.LongTensor, int]
        kernel_size,  # type: int
        dilation,  # type: int
        stride,  # type: int
        poolsize,  # type: int
        padding=None,  # type: Optional[int]
    ):
        # type: (...) -> Union[torch.LongTensor, int]
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        size = size.float() if torch.is_tensor(size) else float(size)
        size = (size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        size = size.floor() if torch.is_tensor(size) else math.floor(size)
        if poolsize:
            size /= poolsize
        if torch.is_tensor(size):
            return size.floor().long()
        else:
            return int(math.floor(size))
