from __future__ import absolute_import

from collections import OrderedDict
from typing import Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from laia.data.padding_collater import PaddedTensor
from laia.models.kws.dortmund_phocnet import build_conv_model, size_after_conv
from laia.nn.image_pooling_sequencer import ImagePoolingSequencer


class DortmundCRNN(torch.nn.Module):
    def __init__(
        self,
        num_outputs,  # type: int
        lstm_hidden_size=128,  # type: int
        lstm_num_layers=1,  # type: int
        sequencer="avgpool-16",  # type: str
        dropout=0.5,  # type: float
    ):
        # type: (...) -> None
        super(DortmundCRNN, self).__init__()
        self._dropout = dropout
        self.conv = build_conv_model()
        self.sequencer = ImagePoolingSequencer(sequencer=sequencer, columnwise=True)
        self.blstm = torch.nn.LSTM(
            input_size=512 * self.sequencer.fix_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = torch.nn.Linear(2 * lstm_hidden_size, num_outputs)

    def dropout(self, x, p=0.5):
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

    def forward(self, x):
        # type: (Union[torch.Tensor, PaddedTensor]) -> Union[torch.Tensor, PackedSequence]
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        x = self.conv(x)
        if xs is not None:
            xs = size_after_conv(xs)
        x = self.sequencer(x if xs is None else PaddedTensor(x, xs))
        x = self.dropout(x, p=self._dropout)
        x, _ = self.blstm(x)
        x = self.dropout(x, p=self._dropout)
        return (
            PackedSequence(self.linear(x), x.batch_sizes)
            if isinstance(x, PackedSequence)
            else self.linear(x)
        )


def convert_old_parameters(params):
    """Convert parameters from the old model to the new one."""
    # type: OrderedDict -> OrderedDict
    new_params = []
    for k, v in params.items():
        if k.startswith("conv"):
            new_params.append(("conv.{}".format(k), v))
        elif k == "linear._module.weight":
            new_params.append(("linear.weight", v))
        elif k == "linear._module.bias":
            new_params.append(("linear.bias", v))
        else:
            new_params.append((k, v))
    return OrderedDict(new_params)
