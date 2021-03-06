import numpy as np
import pytest
import torch

import laia.nn.resnet as resnet
from laia.data import PaddedTensor


def test_basicblock_forward():
    net = resnet.BasicBlock(inplanes=8, planes=8)
    y = net(torch.randn(4, 8, 15, 12))
    assert y.size() == (4, 8, 15, 12)

    net = resnet.BasicBlock(inplanes=3, planes=8)
    y = net(torch.randn(4, 3, 15, 12))
    assert y.size() == (4, 8, 15, 12)

    net = resnet.BasicBlock(inplanes=3, planes=8, stride=2)
    y = net(torch.randn(4, 3, 15, 12))
    assert y.size() == (4, 8, 8, 6)

    net = resnet.BasicBlock(
        inplanes=3, planes=5, stride=2, norm_layer=torch.nn.BatchNorm2d
    )
    y = net(torch.randn(4, 3, 15, 13))
    assert y.size() == (4, 5, 8, 7)


def test_bottleneck_forward():
    net = resnet.Bottleneck(inplanes=8, planes=8)
    y = net(torch.randn(4, 8, 15, 12))
    assert y.size() == (4, 8 * 4, 15, 12)

    net = resnet.Bottleneck(inplanes=3, planes=8)
    y = net(torch.randn(4, 3, 15, 12))
    assert y.size() == (4, 8 * 4, 15, 12)

    net = resnet.Bottleneck(inplanes=3, planes=8, stride=2)
    y = net(torch.randn(4, 3, 15, 12))
    assert y.size() == (4, 8 * 4, 8, 6)

    net = resnet.Bottleneck(
        inplanes=3, planes=5, stride=2, norm_layer=torch.nn.BatchNorm2d
    )
    y = net(torch.randn(4, 3, 15, 13))
    assert y.size() == (4, 5 * 4, 8, 7)


@pytest.mark.parametrize(
    "block,output_dim,extra_groups",
    [(resnet.BasicBlock, 32, 1), (resnet.Bottleneck, 32 * 4, 2)],
)
def test_forward_tensor(block, output_dim, extra_groups):
    options = resnet.ResnetOptions(
        block=block,
        input_channels=1,
        layers=(1, 1, 1, 1),  # Fewer layers/block to be faster.
        stride=(1, 2, 1, 1),
        width_per_group=4,  # Fewer units/group to be faster.
        norm_layer=None,  # No normalization to be faster.
    )
    net = resnet.ResnetConv(options)
    y = net(torch.randn(4, 1, 16, 32))
    assert y.size() == (4, output_dim, 2, 4)

    options = resnet.ResnetOptions(
        block=block,
        input_channels=1,
        root_kernel=3,
        layers=(1, 2, 1, 1),  # Fewer layers/block to be faster.
        stride=(1, 1, 1, 1),
        width_per_group=4,  # Fewer units/group to be faster.
        zero_init_residual=True,
        groups=extra_groups,
        norm_layer=torch.nn.BatchNorm2d,
    )
    net = resnet.ResnetConv(options)
    y = net(torch.randn(3, 1, 17, 19))
    assert y.size() == (3, extra_groups * output_dim, 5, 5)


@pytest.mark.parametrize(
    "block,output_dim,extra_groups",
    [(resnet.BasicBlock, 32, 1), (resnet.Bottleneck, 32 * 4, 2)],
)
def test_forward_padded_tensor(block, output_dim, extra_groups):
    options = resnet.ResnetOptions(
        block=block,
        input_channels=1,
        layers=(1, 1, 1, 1),  # Fewer layers/block to be faster.
        stride=(1, 2, 1, 1),
        width_per_group=4,  # Fewer units/group to be faster.
        norm_layer=None,  # No normalization to be faster.
    )
    net = resnet.ResnetConv(options)
    input_sizes = torch.tensor(
        [[19, 37], [19, 17], [11, 9], [17, 37]], dtype=torch.int64
    )
    y = net(PaddedTensor(data=torch.randn(4, 1, 19, 37), sizes=input_sizes))
    assert isinstance(y, PaddedTensor)
    assert y.data.size() == (4, output_dim, 3, 5)
    expected_sizes = np.asarray([[3, 5], [3, 3], [2, 2], [3, 5]], dtype=np.int64)
    np.testing.assert_array_equal(y.sizes.numpy(), expected_sizes)

    options = resnet.ResnetOptions(
        block=block,
        input_channels=1,
        root_kernel=3,
        layers=(1, 2, 1, 1),  # Fewer layers/block to be faster.
        stride=(1, 1, 1, 1),
        width_per_group=4,  # Fewer units/group to be faster.
        zero_init_residual=True,
        groups=extra_groups,
        norm_layer=torch.nn.BatchNorm2d,
    )
    net = resnet.ResnetConv(options)
    input_sizes = torch.tensor([[17, 19], [5, 3], [17, 11]], dtype=torch.int32)
    y = net(PaddedTensor(data=torch.randn(3, 1, 17, 19), sizes=input_sizes))
    assert y.data.size() == (3, extra_groups * output_dim, 5, 5)
    expected_sizes = np.asarray([[5, 5], [2, 1], [5, 3]], dtype=np.int32)
    np.testing.assert_array_equal(y.sizes.numpy(), expected_sizes)
