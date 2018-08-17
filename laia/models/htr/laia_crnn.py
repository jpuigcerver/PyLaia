from itertools import count
from typing import Sequence, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from laia.common.types import Param2d, ParamNd
from laia.data import PaddedTensor
from laia.models.htr.conv_block import ConvBlock
from laia.nn.image_pooling_sequencer import ImagePoolingSequencer


class LaiaCRNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_output_labels: int,
        cnn_num_features: Sequence[int],
        cnn_kernel_size: Sequence[Param2d],
        cnn_stride: Sequence[Param2d],
        cnn_dilation: Sequence[Param2d],
        cnn_activation: Sequence[nn.Module],
        cnn_poolsize: Sequence[Param2d],
        cnn_dropout: Sequence[float],
        cnn_batchnorm: Sequence[bool],
        image_sequencer: str,
        rnn_units: int,
        rnn_layers: int,
        rnn_dropout: float,
        lin_dropout: float,
        rnn_type: Union[nn.LSTM, nn.GRU, nn.RNN] = nn.LSTM,
        inplace: bool = False,
        vertical_text: bool = False,
        use_masks: bool = False,
    ) -> None:
        super().__init__()
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

    def dropout(
        self, x: Union[torch.Tensor, PaddedTensor, PackedSequence], p: float
    ) -> Union[torch.Tensor, PaddedTensor, PackedSequence]:
        if 0.0 < p < 1.0:
            cls = None
            if isinstance(x, PaddedTensor):
                cls = PaddedTensor
                x, xs = x.data, x.sizes
            elif isinstance(x, PackedSequence):
                cls = PackedSequence
                x, xs = x.data, x.batch_sizes
            d = F.dropout(x, p=p, training=self.training)
            return cls(d, xs) if cls is not None else d
        else:
            return x

    def forward(
        self, x: Union[torch.Tensor, PaddedTensor]
    ) -> Union[torch.Tensor, PackedSequence]:
        x = self.conv(x)
        x = self.sequencer(x)
        x = self.dropout(x, p=self._rnn_dropout)
        x, _ = self.rnn(x)
        x = self.dropout(x, p=self._lin_dropout)
        return (
            PackedSequence(self.linear(x.data), x.batch_sizes)
            if isinstance(x, PackedSequence)
            else self.linear(x)
        )

    @staticmethod
    def get_conv_output_size(
        sizes: Sequence[int],
        cnn_kernel_size: Sequence[ParamNd],
        cnn_stride: Sequence[ParamNd],
        cnn_dilation: Sequence[ParamNd],
        cnn_poolsize: Sequence[ParamNd],
    ) -> List[Union[torch.Tensor, int]]:
        kernel_size, stride, dilation, poolsize = ConvBlock.prepare_dimensional_args(
            cnn_kernel_size, cnn_stride, cnn_dilation, cnn_poolsize, dims=len(sizes)
        )
        return [
            ConvBlock.get_output_size(
                size,
                kernel_size=ks[dim],
                dilation=di[dim],
                stride=st[dim],
                poolsize=ps[dim],
            )
            for dim, size in enumerate(sizes)
            for ks, st, di, ps in zip(kernel_size, stride, dilation, poolsize)
        ]
