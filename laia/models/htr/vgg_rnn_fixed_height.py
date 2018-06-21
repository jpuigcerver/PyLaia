from itertools import count

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from laia.data import PaddedTensor
from laia.models.htr.conv_block import ConvBlock


class VggRnnFixedHeight(nn.Module):
    def __init__(
        self,
        input_height,
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
        rnn_type=nn.LSTM,
    ):
        super(VggRnnFixedHeight, self).__init__()
        self._input_height = input_height
        self._rnn_dropout = rnn_dropout
        self._lin_dropout = lin_dropout
        # Add convolutional blocks, in a VGG style.
        conv_blocks = []
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
            )
            ni = nh
            di_h = di[0] if isinstance(di, (list, tuple)) else di
            ks_h = ks[0] if isinstance(ks, (list, tuple)) else ks
            st_h = st[0] if isinstance(st, (list, tuple)) else st
            ps_h = ps[0] if isinstance(ps, (list, tuple)) else ps
            ps_h = ps_h if ps_h > 1 else None
            input_height = ConvBlock.get_output_size(
                size=input_height,
                dilation=di_h,
                kernel_size=ks_h,
                stride=st_h,
                poolsize=ps_h,
            )
            conv_blocks.append(layer)
        self.conv = Sequential(*conv_blocks)

        self.rnn = rnn_type(
            input_height * ni,
            rnn_units,
            rnn_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.rnn.flatten_parameters()

        self.linear = nn.Linear(2 * rnn_units, num_output_labels)

    def forward(self, x):
        x = self.conv(x)
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        # Note: x shape is N x C x H x W -> W x N x (H * C)
        N, C, H, W = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(W, N, H * C)
        if self._rnn_dropout:
            x = F.dropout(x, self._rnn_dropout, training=self.training)
        if xs:
            x = pack_padded_sequence(x, list(xs.data[:, 1]))
        x, _ = self.rnn(x)
        # Output linear layer
        if xs:
            x, xs = x.data, x.batch_sizes
        if self._lin_dropout:
            x = F.dropout(x, self._lin_dropout, training=self.training)
        x = self.linear(x)
        if xs:
            x = PackedSequence(x, batch_sizes=xs)
        return x
