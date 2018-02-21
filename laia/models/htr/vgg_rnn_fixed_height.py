import re
import torch.nn as nn
import torch.nn.functional as F

from laia.models.htr.conv_block import ConvBlock
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from laia.data import PaddedTensor

class VggRnnFixedHeight(nn.Module):
    def __init__(self, input_height, num_input_channels, num_output_labels,
                 cnn_num_features,
                 cnn_kernel_size,
                 cnn_dilation,
                 cnn_activation,
                 cnn_poolsize,
                 cnn_dropout,
                 cnn_batchnorm,
                 rnn_units,
                 rnn_layers,
                 rnn_dropout,
                 lin_dropout,
                 rnn_type=nn.LSTM):
        super(VggRnnFixedHeight, self).__init__()
        self._input_height = input_height
        self._rnn_dropout = rnn_dropout
        self._lin_dropout = lin_dropout
        # Add convolutional blocks, in a VGG style.
        self._conv_blocks = []
        ni = num_input_channels
        for i, (nh, ks, di, f, ps, dr, bn) in enumerate(
                zip(cnn_num_features, cnn_kernel_size, cnn_dilation,
                    cnn_activation, cnn_poolsize, cnn_dropout, cnn_batchnorm)):
            layer = ConvBlock(ni, nh, kernel_size=ks, dilation=di,
                              activation=f, poolsize=ps, dropout=dr,
                              batchnorm=bn)
            ni = nh
            ps_h = ps[0] if isinstance(ps, (list, tuple)) else ps
            ps_h = ps_h if ps_h > 1 else 1
            input_height = input_height / ps_h
            self.add_module('conv_block%d' % i, layer)
            self._conv_blocks.append(layer)

        rnn = rnn_type(input_height * ni, rnn_units, rnn_layers,
                      dropout=rnn_dropout, bidirectional=True,
                      batch_first=False)
        self.add_module('rnn', rnn)
        self._rnn = rnn

        linear = nn.Linear(2 * rnn_units, num_output_labels)
        self.add_module('linear', linear)
        self._linear = linear

    def forward(self, x):
        is_padded = isinstance(x, PaddedTensor)
        # Check that the input height is the expected (all images in the batch
        # should have the same height).
        assert ((is_padded and x.data.size()[2] == self._input_height) or
                (not is_padded and x.size()[2] == self._input_height))
        for block in self._conv_blocks:
            x = block(x)
        if is_padded:
            x, xs = x.data, x.sizes
        # Note: x shape is N x C x H x W -> W x N x (H * C)
        N, C, H, W = tuple(x.size())
        x = x.permute(3, 0, 1, 2).contiguous().view(W, N, H * C)
        if self._rnn_dropout > 0.0:
            x = F.dropout(x, self._rnn_dropout, training=self.training)
        if is_padded:
            x = pack_padded_sequence(x, list(xs[:,1]))
        x, _ = self._rnn(x)
        # Output linear layer
        if is_padded:
            x, xs = x.data, x.batch_sizes
        if self._lin_dropout > 0.0:
            x = F.dropout(x, self._lin_dropout, training=self.training)
        x = self._linear(x)
        if is_padded:
            x = PackedSequence(data=x, batch_sizes=xs)
        return x
