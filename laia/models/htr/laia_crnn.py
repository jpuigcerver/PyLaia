from itertools import count

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from typing import Sequence, Tuple, Union

from laia.data import PaddedTensor
from laia.models.htr.conv_block import ConvBlock
from laia.nn.image_pooling_sequencer import ImagePoolingSequencer


class LaiaCRNN(nn.Module):
    def __init__(
        self,
        num_input_channels,  # type: int
        num_output_labels,  # type: int
        cnn_num_features,  # type: Sequence[int]
        cnn_kernel_size,  # type: Sequence[int, Tuple[int, int]]
        cnn_stride,  # type: Sequence[int, Tuple[int, int]]
        cnn_dilation,  # type: Sequence[int, Tuple[int, int]]
        cnn_activation,  # type: Sequence[nn.Module]
        cnn_poolsize,  # type: Sequence[int, Tuple[int, int]]
        cnn_dropout,  # type: Sequence[float]
        cnn_batchnorm,  # type: Sequence[bool]
        image_sequencer,  # type: str
        rnn_units,  # type: int
        rnn_layers,  # type: int
        rnn_dropout,  # type: float
        lin_dropout,  # type: float
        rnn_type=nn.LSTM,  # type: Union[nn.LSTM, nn.GRU, nn.RNN]
        inplace=False,  # type: bool
        vertical_text=False,  # type: bool
        use_masks=False,  # type: bool
    ):
        # type: (...) -> None
        super(LaiaCRNN, self).__init__()
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
            conv_blocks.append(
                ConvBlock(
                    in_channels=ni,
                    out_channels=nh,
                    kernel_size=ks,
                    stride=st,
                    dilation=di,
                    activation=f,
                    poolsize=ps,
                    dropout=dr,
                    batchnorm=bn,
                    inplace=inplace,
                    use_masks=use_masks,
                )
            )
            ni = nh
        self.conv = nn.Sequential(*conv_blocks)
        # Add sequencer module to convert an image into a sequence
        self.sequencer = ImagePoolingSequencer(
            sequencer=image_sequencer, columnwise=not vertical_text
        )
        # Add bidirectional rnn
        self.rnn = rnn_type(
            ni * self.sequencer.fix_size,
            rnn_units,
            rnn_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.rnn.flatten_parameters()
        # Add final linear layer
        self.linear = nn.Linear(2 * rnn_units, num_output_labels)

    def dropout(self, x, p):
        if 0.0 < p < 1.0:
            if isinstance(x, PaddedTensor):
                return PaddedTensor(
                    data=F.dropout(x.data, p=p, training=self.training), sizes=x.sizes
                )
            elif isinstance(x, PackedSequence):
                return PackedSequence(
                    data=F.dropout(x.data, p=p, training=self.training),
                    batch_sizes=x.batch_sizes,
                )
            else:
                return F.dropout(x, p=p, training=self.training)
        else:
            return x

    def forward(self, x):
        # type: (Union[Variable, PaddedTensor]) -> Union[Variable, PackedSequence]
        x = self.conv(x)
        x = self.sequencer(x)
        x = self.dropout(x, p=self._rnn_dropout)
        x, _ = self.rnn(x)
        x = self.dropout(x, p=self._lin_dropout)
        return (
            PackedSequence(data=self.linear(x.data), batch_sizes=x.batch_sizes)
            if isinstance(x, PackedSequence)
            else self.linear(x)
        )

    @staticmethod
    def get_conv_output_size(
        size,  # type: Tuple[int, int]
        cnn_kernel_size,  # type: Sequence[Union[int, Tuple[int, int]]]
        cnn_stride,  # type: Sequence[Union[int, Tuple[int, int]]]
        cnn_dilation,  # type: Sequence[Union[int, Tuple[int, int]]]
        cnn_poolsize,  # type: Sequence[Union[int, Tuple[int, int]]]
    ):
        size_h, size_w = size
        for ks, st, di, ps in zip(
            cnn_kernel_size, cnn_stride, cnn_dilation, cnn_poolsize
        ):
            size_h = ConvBlock.get_output_size(
                size_h, kernel_size=ks[0], dilation=di[0], stride=st[0], poolsize=ps[0]
            )
            size_w = ConvBlock.get_output_size(
                size_w, kernel_size=ks[1], dilation=di[1], stride=st[1], poolsize=ps[1]
            )
        return size_h, size_w
