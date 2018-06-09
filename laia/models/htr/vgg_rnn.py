from itertools import count

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from nnutils_pytorch import mask_image_from_size
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from laia.data import PaddedTensor
from laia.models.htr.conv_block import ConvBlock


def collapse_mean(x, xs):
    if xs is None:
        return x.mean(dim=2, keepdim=True)
    else:
        x = mask_image_from_size(mask_value=0, inplace=True)(x, xs)
        return x.sum(dim=2, keepdim=True) / xs[:, 0]


def collapse_max(x, xs):
    if xs is not None:
        x = mask_image_from_size(mask_value=np.NINF, inplace=True)(x, xs)
    return x.max(dim=2, keepdim=True)[0]


def collapse_sum(x, xs):
    if xs is not None:
        x = mask_image_from_size(mask_value=0, inplace=True)(x, xs)
    return x.sum(dim=2, keepdim=True)


def collapse_min(x, xs):
    if xs is not None:
        x = mask_image_from_size(mask_value=np.inf, inplace=True)(x, xs)
    return x.min(dim=2, keepdim=True)[0]


class VggRnn(nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_output_labels,
        cnn_num_features,
        cnn_kernel_size,
        cnn_stride,
        cnn_dilation,
        cnn_activation,
        cnn_poolsize,
        cnn_dropout,
        cnn_batchnorm,
        rnn_units,
        rnn_layers,
        rnn_dropout,
        lin_dropout,
        collapse="mean",  # Can be a function as well
        inplace=False,
    ):
        super(VggRnn, self).__init__()
        self._rnn_dropout = rnn_dropout
        self._lin_dropout = lin_dropout
        # Add convolutional blocks, in a VGG style.
        self._conv_blocks = []
        ni = num_input_channels
        for i, nh, ks, st, di, f, ps, dr, bn in zip(
            count(),
            cnn_num_features,
            cnn_kernel_size,
            cnn_stride,
            cnn_dilation,
            cnn_activation,
            cnn_poolsize,
            cnn_dropout,
            cnn_batchnorm,
        ):
            layer = ConvBlock(
                ni,
                nh,
                kernel_size=ks,
                stride=st,
                dilation=di,
                activation=f,
                poolsize=ps,
                dropout=dr,
                batchnorm=bn,
                inplace=inplace,
            )
            ni = nh
            self.add_module("conv_block{}".format(i), layer)
            self._conv_blocks.append(layer)

        rnn = nn.LSTM(
            ni,
            rnn_units,
            rnn_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.add_module("rnn", rnn)
        self._rnn = rnn

        linear = nn.Linear(2 * rnn_units, num_output_labels)
        self.add_module("linear", linear)
        self._linear = linear

        collapse_funcs = {
            "sum": collapse_sum,
            "max": collapse_max,
            "min": collapse_min,
            "mean": collapse_mean,
        }
        if callable(collapse):
            self._collapse = collapse
        else:
            self._collapse = collapse_funcs.get(collapse.lower(), None)
            if self._collapse is None:
                raise NotImplementedError

    def forward(self, x):
        is_padded = isinstance(x, PaddedTensor)
        for block in self._conv_blocks:
            x = block(x)
        if is_padded:
            x, xs = x.data, x.sizes
        # Note: x shape is N x C x H x W
        x = self._collapse(x, xs if is_padded else None)
        # Note: x shape is typically N x C x 1 x W
        N, C, H, W = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(W, N, C * H)
        if self._rnn_dropout:
            x = F.dropout(x, self._rnn_dropout, training=self.training)
        if is_padded:
            x = pack_padded_sequence(x, list(xs[:, 1]))
        self._rnn.flatten_parameters()
        x, _ = self._rnn(x)
        # Output linear layer
        if is_padded:
            x, xs = x.data, x.batch_sizes
        if self._lin_dropout:
            x = F.dropout(x, self._lin_dropout, training=self.training)
        x = self._linear(x)
        if is_padded:
            x = PackedSequence(x, batch_sizes=xs)
        return x
