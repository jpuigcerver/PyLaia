from itertools import count
from typing import List, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from laia.common.types import Param2d, ParamNd
from laia.data import PaddedTensor
from laia.models.htr import ConvBlock
from laia.nn import ImagePoolingSequencer


class LaiaCRNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_output_labels: int,
        cnn_num_features: Sequence[int],
        cnn_kernel_size: Sequence[Param2d],
        cnn_stride: Sequence[Param2d],
        cnn_dilation: Sequence[Param2d],
        cnn_activation: Sequence[Type[nn.Module]],
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
        return x

    def forward(
        self, x: Union[torch.Tensor, PaddedTensor]
    ) -> Union[torch.Tensor, PackedSequence]:
        if isinstance(x, PaddedTensor):
            xs = self.get_self_conv_output_size(x.sizes)
            err_indices = [i for i, x in enumerate((xs < 1).any(1)) if x]
            if err_indices:
                raise ValueError(
                    f"The images at batch indices {err_indices} "
                    f"with sizes {x.sizes[err_indices].tolist()} "
                    f"would produce invalid output sizes {xs[err_indices].tolist()}"
                )
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
        size: Param2d,
        cnn_kernel_size: Sequence[ParamNd],
        cnn_stride: Sequence[ParamNd],
        cnn_dilation: Sequence[ParamNd],
        cnn_poolsize: Sequence[ParamNd],
    ) -> Tuple[Union[torch.LongTensor, int]]:
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

    def get_self_conv_output_size(
        self, size: Param2d
    ) -> List[Union[torch.Tensor, int]]:
        xs = size.clone()
        for l in self.conv:
            xs = l.get_batch_output_size(xs)
        return xs
