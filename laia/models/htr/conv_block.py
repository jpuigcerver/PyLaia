import math
from typing import Optional, Union, List, Any, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from laia.common.types import Param2d
from laia.data import PaddedTensor

try:
    from laia.nn.mask_image_from_size import mask_image_from_size
except ImportError:
    import warnings

    mask_image_from_size = None


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Param2d = 3,
        stride: Param2d = 1,
        dilation: Param2d = 1,
        activation: Optional[nn.Module] = nn.LeakyReLU,
        poolsize: Param2d = 0,
        dropout: Optional[float] = None,
        batchnorm: bool = False,
        inplace: bool = False,
        use_masks: bool = False,
    ) -> None:
        super().__init__()
        ks, st, di, ps = ConvBlock.prepare_dimensional_args(
            kernel_size, stride, dilation, poolsize
        )

        if ps[0] * ps[1] < 2:
            ps = None

        if use_masks and mask_image_from_size is None:
            warnings.warn(
                "nnutils does not seem to be installed, masking cannot be used"
            )
            use_masks = False

        self.dropout = dropout
        self.in_channels = in_channels
        self.use_masks = use_masks
        self.poolsize = ps

        # Add Conv2d layer (compute padding to perform a full convolution).
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            ks,
            stride=st,
            padding=tuple((ks[dim] - 1) // 2 * di[dim] for dim in (0, 1)),
            dilation=di,
            # Note: If batchnorm is used, the bias does not
            # affect the output of the unit.
            bias=not batchnorm,
        )

        # Add Batch normalization
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # Activation function must support inplace operations.
        self.activation = activation(inplace=inplace) if activation else None

        # Add maxpool layer
        self.pool = nn.MaxPool2d(ps) if self.poolsize else None

    @staticmethod
    def prepare_dimensional_args(*args: Any, dims: int = 2) -> List[Tuple]:
        return [
            tuple(arg) if isinstance(arg, (list, tuple)) else (arg,) * dims
            for arg in args
        ]

    def forward(self, x: Union[Tensor, PaddedTensor]) -> Union[Tensor, PaddedTensor]:
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
            x = mask_image_from_size(x, batch_sizes=xs, mask_value=0)

        if self.batchnorm:
            x = self.batchnorm(x)

        if self.activation:
            x = self.activation(x)

        if self.use_masks:
            x = mask_image_from_size(x, batch_sizes=xs, mask_value=0)

        if self.pool:
            x = self.pool(x)

        return x if xs is None else PaddedTensor(x, self.get_output_batch_size(xs))

    def get_output_batch_size(self, xs: torch.Tensor):
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
        return ys

    @staticmethod
    def get_output_size(
        size: Union[torch.Tensor, int],
        kernel_size: int,
        dilation: int,
        stride: int,
        poolsize: int,
        padding: Optional[int] = None,
    ) -> Union[torch.Tensor, int]:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        size = size.float() if isinstance(size, torch.Tensor) else float(size)
        size = (size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        size = size.floor() if isinstance(size, torch.Tensor) else math.floor(size)
        if poolsize:
            size /= poolsize
        return (
            size.floor().long()
            if isinstance(size, torch.Tensor)
            else int(math.floor(size))
        )
