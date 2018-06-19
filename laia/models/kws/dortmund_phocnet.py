from __future__ import absolute_import

import math
import operator
from collections import OrderedDict
from functools import reduce

import torch
from torch.autograd import Variable
from typing import Union, Sequence, Optional

from laia.data import PaddedTensor
from laia.nn.pyramid_maxpool_2d import PyramidMaxPool2d
from laia.nn.temporal_pyramid_maxpool_2d import TemporalPyramidMaxPool2d


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def build_conv_model(unittest=False):
    model = torch.nn.Sequential(
        OrderedDict(
            [
                # conv1_1
                ("conv1_1", torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)),
                ("relu1_1", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv1_2
                ("conv1_2", torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)),
                ("relu1_2", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                ("maxpool1", torch.nn.MaxPool2d(2, ceil_mode=True)),
                # conv2_1
                ("conv2_1", torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)),
                ("relu2_1", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv2_2
                ("conv2_2", torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)),
                ("relu2_2", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                ("maxpool2", torch.nn.MaxPool2d(2, ceil_mode=True)),
                # conv3_1
                ("conv3_1", torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                ("relu3_1", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv3_2
                ("conv3_2", torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ("relu3_2", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv3_3
                ("conv3_3", torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ("relu3_3", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv3_4
                ("conv3_4", torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ("relu3_4", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv3_5
                ("conv3_5", torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ("relu3_5", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv3_6
                ("conv3_6", torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ("relu3_6", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv4_1
                ("conv4_1", torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)),
                ("relu4_1", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv4_2
                ("conv4_2", torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ("relu4_2", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                # conv4_3
                ("conv4_3", torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ("relu4_3", Identity() if unittest else torch.nn.ReLU(inplace=True)),
            ]
        )
    )
    return model


def size_after_conv(xs):
    # type: (Union[torch.Tensor, Variable]) -> object
    xs = xs.float()
    xs = torch.ceil(xs / 2.0)
    xs = torch.ceil(xs / 2.0)
    return xs.long()


class DortmundPHOCNet(torch.nn.Module):
    def __init__(
        self, phoc_size, tpp_levels=range(1, 6), spp_levels=None, unittest=False
    ):
        # type: (int, Optional[Sequence[int]], Optional[Sequence[int]], bool) -> None
        super(DortmundPHOCNet, self).__init__()
        assert tpp_levels or spp_levels
        if tpp_levels is None:
            tpp_levels = []
        if spp_levels is None:
            spp_levels = []
        self.conv = build_conv_model(unittest=unittest)
        self.tpp = TemporalPyramidMaxPool2d(levels=tpp_levels) if tpp_levels else None
        self.spp = PyramidMaxPool2d(levels=spp_levels) if spp_levels else None

        # Size after the temporal and spatial pooling layers
        fc_input_dim = 512 * (
            sum(tpp_levels) + sum([2 ** (lv - 1) * 2 ** (lv - 1) for lv in spp_levels])
        )
        self.fc = torch.nn.Sequential(
            OrderedDict(
                [
                    ("fc6", torch.nn.Linear(fc_input_dim, 4096)),
                    ("relu6", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                    ("drop6", torch.nn.Dropout(p=0 if unittest else 0.5)),
                    ("fc7", torch.nn.Linear(4096, 4096)),
                    ("relu7", Identity() if unittest else torch.nn.ReLU(inplace=True)),
                    ("drop7", torch.nn.Dropout(p=0 if unittest else 0.5)),
                    ("fc8", torch.nn.Linear(4096, phoc_size)),
                ]
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters as Caffe does
        for name, param in self.named_parameters():
            if name[-5:] == ".bias":
                # Initialize bias to 0
                param.data.fill_(0)
            else:
                # compute fan in
                fan_in = reduce(operator.mul, param.data.size()[1:])
                param.data.normal_(mean=0, std=math.sqrt(2.0 / fan_in))
        return self

    def forward(self, x):
        # type: (Union[Variable, PaddedTensor]) -> Variable
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        x = self.conv(x)
        if xs is not None:
            xs = size_after_conv(xs)
            x = PaddedTensor(data=x, sizes=xs)
        if self.tpp and self.spp:
            x = torch.cat([self.tpp(x), self.spp(x)], dim=1)
        elif self.tpp:
            x = self.tpp(x)
        else:
            x = self.spp(x)
        return self.fc(x)


def convert_old_parameters(params):
    """Convert parameters from the old model to the new one."""
    # type: OrderedDict -> OrderedDict
    new_params = []
    for k, v in params.items():
        if k[:4] == "conv":
            new_params.append(("conv.{}".format(k), v))
        elif k[:2] == "fc":
            new_params.append(("fc.{}".format(k), v))
        else:
            new_params.append((k, v))
    return OrderedDict(new_params)
