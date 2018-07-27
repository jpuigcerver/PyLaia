from __future__ import absolute_import

import unittest

import torch

from laia.data import PaddedTensor
from laia.nn import MaskImageFromSize

try:
    from laia.nn.mask_image_from_size import MaskImageFromSize
except ImportError:
    MaskImageFromSize = None


@unittest.skipIf(MaskImageFromSize is None, "nnutils does not seem installed")
class MaskImageFromSizeTest(unittest.TestCase):
    def test_tensor(self):
        x = torch.randn(3, 5, 7, 9, requires_grad=True)
        layer = MaskImageFromSize(mask_value=-99)
        y = layer(x)
        torch.testing.assert_allclose(y, x)
        dx, = torch.autograd.grad([torch.sum(y)], [x])
        torch.testing.assert_allclose(dx, torch.ones(3, 5, 7, 9))

    def test_padded_tensor(self):
        x = torch.randn(3, 5, 7, 9, requires_grad=True)
        xs = torch.tensor([[1, 1], [3, 9], [7, 5]])
        layer = MaskImageFromSize(mask_value=-99)
        y = layer(PaddedTensor(x, xs))
        # Expected output
        expected_y = x.clone()
        expected_y[0, :, 1:, :] = -99
        expected_y[0, :, :, 1:] = -99
        expected_y[1, :, 3:, :] = -99
        expected_y[2, :, :, 5:] = -99
        # Check expected output
        torch.testing.assert_allclose(y.data, expected_y)

        # Test backward pass (mask with 0, so that we can sum without masking)
        layer = MaskImageFromSize(mask_value=0)
        y = layer(PaddedTensor(x, xs))
        dx, = torch.autograd.grad([torch.sum(y.data)], [x])
        # Expected output
        expected_dx = torch.ones(3, 5, 7, 9)
        expected_dx[0, :, 1:, :] = 0
        expected_dx[0, :, :, 1:] = 0
        expected_dx[1, :, 3:, :] = 0
        expected_dx[2, :, :, 5:] = 0
        # Check expected output
        torch.testing.assert_allclose(dx, expected_dx)


if __name__ == "__main__":
    unittest.main()
