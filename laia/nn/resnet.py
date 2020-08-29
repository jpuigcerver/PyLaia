from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from laia.data import PaddedTensor
from laia.data.padding_collater import transform_output


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample(inplanes, planes, expansion, stride=1, norm_layer=None):
    if stride != 1 or inplanes != planes * expansion:
        downsample = [conv1x1(inplanes, planes * expansion, stride)]
        if norm_layer is not None:
            downsample.append(norm_layer(planes * expansion))
        return nn.Sequential(*downsample)
    else:
        return None


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super().__init__()
        if groups != 1:
            raise ValueError("BasicBlock only supports groups=1")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = None if norm_layer is None else norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = None if norm_layer is None else norm_layer(planes)
        self.downsample = downsample(
            inplanes, planes, self.expansion, stride, norm_layer
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = None if norm_layer is None else norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = None if norm_layer is None else norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = None if norm_layer is None else norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample(
            inplanes, planes, self.expansion, stride, norm_layer
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResnetOptions:
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        input_channels: int = 1,
        root_kernel: int = 5,
        layers: Sequence[int] = (2, 2, 2, 2),
        stride: Sequence[int] = (1, 2, 1, 1),
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):

        if len(layers) != 4:
            raise ValueError("The length of layers should be 4")

        if len(stride) != 4:
            raise ValueError("The length of stride should be 4")

        if root_kernel not in [3, 5, 7]:
            raise ValueError("The root_kernel must be 3, 5 or 7")

        self._input_channels = input_channels
        self._root_kernel = root_kernel
        self._root_padding = {3: 1, 5: 2, 7: 3}[root_kernel]
        self._block = block
        self._layers = layers
        self._stride = stride
        self._zero_init_residual = zero_init_residual
        self._groups = groups
        self._width_per_group = width_per_group
        self._norm_layer = norm_layer

        self._planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def root_kernel(self):
        return self._root_kernel

    @property
    def root_padding(self):
        return self._root_padding

    @property
    def block(self):
        return self._block

    @property
    def layers(self):
        return self._layers

    @property
    def stride(self):
        return self._stride

    @property
    def zero_init_residual(self):
        return self._zero_init_residual

    @property
    def groups(self):
        return self._groups

    @property
    def width_per_group(self):
        return self._width_per_group

    @property
    def norm_layer(self):
        return self._norm_layer

    @property
    def planes(self):
        return self._planes


class ResnetConv(nn.Module):
    def __init__(self, options: ResnetOptions):
        super().__init__()

        self.inplanes = options.planes[0]
        self.conv1 = nn.Conv2d(
            in_channels=options.input_channels,
            out_channels=options.planes[0],
            kernel_size=options.root_kernel,
            stride=2,
            padding=options.root_padding,
            bias=False,
        )
        if options.norm_layer is None:
            self.bn1 = None
        else:
            self.bn1 = options.norm_layer(options.planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(4):
            setattr(
                self,
                f"layer{i + 1}",
                self._make_layer(
                    block=options.block,
                    planes=options.planes[i],
                    blocks=options.layers[i],
                    stride=options.stride[i],
                    groups=options.groups,
                    norm_layer=options.norm_layer,
                ),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if options.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self._options = options

    @property
    def options(self):
        return self._options

    def _make_layer(self, block, planes, blocks, stride, groups, norm_layer):
        layers = [block(self.inplanes, planes, stride, groups, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=groups, norm_layer=norm_layer)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x, xs = transform_output(x)
        assert x.size(1) == self._options.input_channels, (
            f"Input image depth ({x.size(1)}) does not match "
            f"the expected ({self._options.input_channels})"
        )

        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if xs is None:
            return x
        else:
            return PaddedTensor(x, self.get_output_batch_size(xs))

    def get_output_batch_size(self, xs):
        return self.get_output_size(xs, self._options)

    @staticmethod
    def get_output_size(
        size: Union[torch.Tensor, int], options: ResnetOptions
    ) -> Union[torch.Tensor, int]:
        # (size + 2 * padding - dilation * (kernel - 1) + stride - 1)  // stride
        size = (size + 1) // 2  # After self.conv1
        size = (size + 1) // 2  # After self.maxpool
        for i in range(4):  # After layer i
            size = (size + options.stride[i] - 1) // options.stride[i]
        return size
