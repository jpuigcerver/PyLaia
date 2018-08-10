from typing import Union, Tuple, Callable, Sequence, Optional

import torch
from torch.nn.functional import dropout
from torch.nn.utils.rnn import PackedSequence

from laia.data.padding_collater import PaddedTensor
from laia.nn.image_pooling_sequencer import ImagePoolingSequencer

Size = Union[int, Tuple[int, int]]
Module = Callable[..., torch.nn.Module]


class GatedConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, activation, stride=1, padding=0
    ):
        # type: (int, int, Size, Module, Size, Size) -> None
        super(GatedConv2d, self).__init__()

        conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        conv2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pregate = torch.nn.Sequential(conv1, activation) if activation else conv1
        self.gate = torch.nn.Sequential(conv2, torch.nn.Sigmoid())

    def forward(self, x):
        x = self.pregate(x)
        y = self.gate(x)
        return x * y


class GatedEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # type: int
        features,  # type: Sequence[int]
        kernel_sizes,  # type: Sequence[Size]
        add_gating,  # type: Sequence[bool]
        strides=1,  # type: Sequence[Size]
        activation=torch.nn.Tanh,
        # type: Union[Module, Sequence[Module]]
        poolsize=None,  # type: Optional[Sequence[Size]]
        inplace=True,  # type: bool
    ):
        # type: (...) -> None
        super(GatedEncoder, self).__init__()
        assert isinstance(features, (tuple, list))
        assert isinstance(kernel_sizes, (tuple, list))
        assert isinstance(add_gating, (tuple, list))
        assert len(features) > 0
        assert len(kernel_sizes) > 0
        n = len(features)

        if len(kernel_sizes) < n:
            kernel_sizes = kernel_sizes + [kernel_sizes[-1]] * (n - len(kernel_sizes))

        if len(add_gating) < n:
            add_gating = add_gating + [False] * (n - len(add_gating))

        if isinstance(strides, (tuple, list)):
            if len(strides) < n:
                strides = strides + [1] * (n - len(strides))
        else:
            strides = [strides] * n

        if isinstance(activation, (tuple, list)):
            assert len(activation) == n
        else:
            activation = [activation] * n

        if isinstance(poolsize, (tuple, list)):
            if len(poolsize) < n:
                poolsize = poolsize + [None] * (n - len(poolsize))
        else:
            poolsize = [poolsize] * n

        if isinstance(inplace, (tuple, list)):
            if len(inplace) < n:
                inplace = inplace + [True] * (n - len(inplace))
        else:
            inplace = [inplace] * n

        self._conv_sizes = []
        self._conv_strides = []
        self._pool_sizes = []
        layers = []
        for i, (n, k, s, f, g, p) in enumerate(
            zip(features, kernel_sizes, strides, activation, add_gating, poolsize)
        ):
            if not isinstance(k, (tuple, list)):
                k = k, k
            if not isinstance(s, (tuple, list)):
                s = s, s
            if not isinstance(p, (tuple, list)) and p is not None:
                p = p, p

            self._conv_sizes.append(k)
            self._conv_strides.append(s)

            if add_gating:
                layers.append(
                    GatedConv2d(
                        in_channels=in_channels,
                        out_channels=n,
                        kernel_size=k,
                        activation=f(inplace=inplace),
                        stride=s,
                    )
                )
            else:
                layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=n,
                            kernel_size=k,
                            stride=s,
                        ),
                        f(inplace=inplace),
                    )
                )

            if p is not None and p[0] > 1 and p[1] > 1:
                layers.append(torch.nn.MaxPool2d(kernel_size=p))
                self._pool_sizes.append(p)
            else:
                self._pool_sizes.append(None)

        self.blocks = torch.nn.Sequential(*layers)

    def _compute_output_size(self, xs):
        h, w = xs[:, 0], xs[:, 1]
        for k, s, p in zip(self._conv_sizes, self._conv_strides, self._pool_sizes):
            h = (h - k[0] - 1) / s[0] + 1
            w = (w - k[1] - 1) / s[1] + 1
            if p:
                h = (h - p[0] - 1) / p[0] + 1
                w = (w - p[1] - 1) / p[1] + 1

        return torch.stack((h, w), dim=1)

    def forward(self, x):
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        y = self.blocks(x)
        return y if xs is None else PaddedTensor(y, self._compute_output_size(xs))


class RNNDecoder(torch.nn.Module):
    def __init__(
        self,
        input_size,  # type: int
        num_outputs,  # type: int
        rnn_hidden_size,  # type: int
        rnn_num_layers,  # type: int
        rnn_type=torch.nn.LSTM,  # type: Module
        bidirectional=True,  # type: bool
        dropout_p=0.0,  # type: float
    ):
        # type: (...) -> None
        super(RNNDecoder, self).__init__()
        self._dropout = dropout_p
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=dropout_p,
        )
        self.linear = torch.nn.Linear(
            2 * rnn_hidden_size if bidirectional else rnn_hidden_size, num_outputs
        )

    def dropout(self, x):
        if self._dropout > 0.0:
            if isinstance(x, PackedSequence):
                x = x.data
            x = dropout(x, self._dropout, self.training)
        return x

    def forward(self, x):
        x = self.dropout(x)
        x = self.rnn(x)
        x = self.dropout(x)
        if isinstance(x, PackedSequence):
            x = x.data
        x = self.linear(x)
        return x


class GatedCRNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # type: int
        num_outputs,  # type: int
        cnn_num_features,  # type: Sequence[int]
        cnn_kernel_sizes,  # type: Sequence[Size]
        cnn_add_gating,  # type: Sequence[bool]
        cnn_strides=1,  # type: Sequence[Size]
        cnn_poolsize=None,  # type: Optional[Sequence[bool]]
        cnn_activation=torch.nn.Tanh,
        # type: Union[Module, Sequence[Module]]
        cnn_use_inplace=True,  # type: bool
        sequencer="maxpool-1",  # type: str
        columnwise=True,  # type: bool
        rnn_hidden_size=128,  # type: int
        rnn_num_layers=2,  # type: int
        rnn_type=torch.nn.LSTM,  # type: Module
        rnn_bidirectional=True,  # type: bool
        rnn_dropout=0.0,  # type: float
    ):
        # type: (...) -> None
        super(GatedCRNN, self).__init__()

        self.encoder = GatedEncoder(
            in_channels=in_channels,
            features=cnn_num_features,
            kernel_sizes=cnn_kernel_sizes,
            add_gating=cnn_add_gating,
            strides=cnn_strides,
            activation=cnn_activation,
            poolsize=cnn_poolsize,
            inplace=cnn_use_inplace,
        )

        self.sequencer = ImagePoolingSequencer(sequencer, columnwise=columnwise)

        self.decoder = RNNDecoder(
            input_size=self.sequencer.fix_size * cnn_num_features[-1],
            num_outputs=num_outputs,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            bidirectional=rnn_bidirectional,
            dropout_p=rnn_dropout,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.sequencer(x)
        return x.decoder(x)
