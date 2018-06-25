from __future__ import absolute_import
from __future__ import division

import unittest

import numpy as np
import torch
from torch.autograd import Variable

from laia.data import PaddedTensor
from laia.models.htr.conv_block import ConvBlock
from laia.models.htr.testing_utils import generate_backprop_floating_point_tests


class ConvBlockTest(unittest.TestCase):
    def test_output_size(self):
        m = ConvBlock(4, 5, kernel_size=3, stride=1, dilation=1, poolsize=2)
        x = torch.randn(3, 4, 11, 13)
        y = m(Variable(x))
        ys = tuple(y.size())
        self.assertTupleEqual(ys, (3, 5, 11 // 2, 13 // 2))

    def test_output_size_padded_tensor(self):
        m = ConvBlock(4, 5, kernel_size=3, stride=1, dilation=1, poolsize=2)
        x = torch.randn(3, 4, 11, 13)
        y = m(
            PaddedTensor(
                Variable(x), Variable(torch.LongTensor([[11, 13], [10, 12], [3, 2]]))
            )
        )
        ys = y.sizes.data.tolist()
        self.assertListEqual(
            ys, [[11 // 2, 13 // 2], [10 // 2, 12 // 2], [3 // 2, 2 // 2]]
        )

    def test_output_size_no_pool(self):
        m = ConvBlock(4, 5, poolsize=0)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(Variable(x), Variable(torch.LongTensor([[11, 13]]))))
        ys = y.sizes.data.tolist()
        ys2 = list(y.data.size())[2:]
        self.assertListEqual(ys, [[11, 13]])
        self.assertListEqual(ys2, [11, 13])

    def test_output_size_poolsize(self):
        m = ConvBlock(4, 5, poolsize=3)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(Variable(x), Variable(torch.LongTensor([[11, 13]]))))
        ys = y.sizes.data.tolist()
        ys2 = list(y.data.size())[2:]
        self.assertListEqual(ys, [[11 // 3, 13 // 3]])
        self.assertListEqual(ys2, [11 // 3, 13 // 3])

    def test_output_size_dilation(self):
        # Note: padding should be added automatically to have the same output size
        m = ConvBlock(4, 5, dilation=3)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(Variable(x), Variable(torch.LongTensor([[11, 13]]))))
        ys = y.sizes.data.tolist()
        ys2 = list(y.data.size())[2:]
        self.assertListEqual(ys, [[11, 13]])
        self.assertListEqual(ys2, [11, 13])

    def test_output_size_stride(self):
        m = ConvBlock(4, 5, stride=2)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(Variable(x), Variable(torch.LongTensor([[11, 13]]))))
        ys = y.sizes.data.tolist()
        ys2 = list(y.data.size())[2:]
        self.assertListEqual(ys, [[11 // 2 + 1, 13 // 2 + 1]])
        self.assertListEqual(ys2, [11 // 2 + 1, 13 // 2 + 1])

    def test_masking(self):
        m = ConvBlock(1, 1, activation=None, use_masks=True)
        # Reset parameters so that the operation does nothing
        for name, param in m.named_parameters():
            param.data.zero_()
            if name == "conv.weight":
                param.data[:, :, 1, 1] = 1

        x = torch.randn(3, 1, 11, 13)
        y = m(
            PaddedTensor(
                Variable(x), Variable(torch.LongTensor([[11, 13], [10, 12], [3, 2]]))
            )
        ).data.data
        x = x.numpy()
        y = y.numpy()

        # Check sample 1
        np.testing.assert_almost_equal(y[0, :, :, :], x[0, :, :, :])
        # Check sample 2
        np.testing.assert_almost_equal(y[1, :, :10, :12], x[1, :, :10, :12])
        np.testing.assert_almost_equal(y[1, :, 10:, :], np.zeros((1, 1, 13)))
        np.testing.assert_almost_equal(y[1, :, :, 12:], np.zeros((1, 11, 1)))
        # Check sample 3
        np.testing.assert_almost_equal(y[2, :, :3, :2], x[2, :, :3, :2])
        np.testing.assert_almost_equal(y[2, :, 3:, :], np.zeros((1, 8, 13)))
        np.testing.assert_almost_equal(y[2, :, :, 2:], np.zeros((1, 11, 11)))


def cost_function(y):
    return y.sum()


def padded_cost_function(padded_y):
    y, ys = padded_y.data, padded_y.sizes.data
    cost = 0
    for y_i, ys_i in zip(y, ys):
        cost = 0 + y_i[:, : ys_i[0], : ys_i[1]].sum()
    return cost


# Add some tests to make sure that the backprop is working correctly.
# Note: this only checks that the gradient w.r.t. all layers is different from zero.
generate_backprop_floating_point_tests(
    ConvBlockTest,
    tests=[
        (
            "backprob_{}_{}_default",
            dict(
                module=ConvBlock,
                module_kwargs=dict(in_channels=3, out_channels=5),
                batch_data=torch.randn(2, 3, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=padded_cost_function,
            ),
        ),
        (
            "backprob_{}_{}_batchnorm",
            dict(
                module=ConvBlock,
                module_kwargs=dict(in_channels=3, out_channels=5, batchnorm=True),
                batch_data=torch.randn(2, 3, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=padded_cost_function,
            ),
        ),
        (
            "backprob_{}_{}_dropout",
            dict(
                module=ConvBlock,
                module_kwargs=dict(in_channels=3, out_channels=5, dropout=0.3),
                batch_data=torch.randn(2, 3, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=padded_cost_function,
            ),
        ),
        (
            "backprob_{}_{}_maxpool",
            dict(
                module=ConvBlock,
                module_kwargs=dict(in_channels=3, out_channels=5, poolsize=2),
                batch_data=torch.randn(2, 3, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=padded_cost_function,
            ),
        ),
        (
            "backprob_{}_{}_use_masks",
            dict(
                module=ConvBlock,
                module_kwargs=dict(in_channels=3, out_channels=5, use_masks=True),
                batch_data=torch.randn(2, 3, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=padded_cost_function,
            ),
        ),
        (
            "backprob_{}_{}_inplace",
            dict(
                module=ConvBlock,
                module_kwargs=dict(in_channels=3, out_channels=5, inplace=True),
                batch_data=torch.randn(2, 3, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=padded_cost_function,
            ),
        ),
    ],
)

if __name__ == "__main__":
    unittest.main()
