from __future__ import absolute_import

import unittest

import torch

from laia.data import PaddedTensor
from laia.nn.resnet import ResnetConv2dBlock


class ResnetConv2dBlockTest(unittest.TestCase):
    def test_equal_planes(self):
        x = torch.randn(4, 16, 8, 12, requires_grad=True)
        layer = ResnetConv2dBlock(16, 16)
        y = layer(x)
        self.assertEqual(x.size(), y.size())
        dx, = torch.autograd.grad([torch.sum(y)], [x])
        self.assertEqual(x.size(), dx.size())

    def test_diff_planes(self):
        x = torch.randn(4, 3, 8, 12, requires_grad=True)
        layer = ResnetConv2dBlock(3, 16)
        y = layer(x)
        self.assertEqual((4, 16, 8, 12), y.size())
        dx, = torch.autograd.grad([torch.sum(y)], [x])
        self.assertEqual(x.size(), dx.size())

    def test_padded_tensor(self):
        x = torch.randn(4, 16, 48, 32, requires_grad=True)
        xs = torch.tensor([[30, 30], [40, 32], [10, 32], [48, 10]])
        x = PaddedTensor(x, xs)
        layer = ResnetConv2dBlock(16, 32)
        y = layer(x)
        self.assertEqual((4, 32, 48, 32), y.data.size())
        self.assertTrue(torch.equal(xs, y.sizes))


if __name__ == "__main__":
    unittest.main()
