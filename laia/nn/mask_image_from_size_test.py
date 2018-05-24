from __future__ import absolute_import

from laia.data import PaddedTensor
from laia.nn import MaskImageFromSize

import numpy as np
import torch
from torch.autograd import Variable
import unittest


class MaskImageFromSizeTest(unittest.TestCase):

    def test_tensor(self):
        x = Variable(torch.randn(3, 5, 7, 9), requires_grad=True)
        layer = MaskImageFromSize(mask_value=-99)
        y = layer(x)
        np.testing.assert_allclose(x.data, y.data)
        dx, = torch.autograd.grad([torch.sum(y)], [x])
        np.testing.assert_allclose(np.ones((3, 5, 7, 9)), dx.data)

    def test_padded_tensor(self):
        x = Variable(torch.randn(3, 5, 7, 9), requires_grad=True)
        xs = Variable(torch.LongTensor([[1, 1], [3, 9], [7, 5]]))
        layer = MaskImageFromSize(mask_value=-99)
        y = layer(PaddedTensor(data=x, sizes=xs))
        # Expected output
        expected_y = x.data.clone()
        expected_y[0, :, 1:, :] = -99
        expected_y[0, :, :, 1:] = -99
        expected_y[1, :, 3:, :] = -99
        expected_y[2, :, :, 5:] = -99
        # Check expected output
        np.testing.assert_allclose(expected_y, y.data.data)

        # Test backward pass (mask with 0, so that we can sum without masking)
        layer = MaskImageFromSize(mask_value=0)
        y = layer(PaddedTensor(data=x, sizes=xs))
        dx, = torch.autograd.grad([torch.sum(y.data)], [x])
        # Expected output
        expected_dx = torch.ones(3, 5, 7, 9)
        expected_dx[0, :, 1:, :] = 0
        expected_dx[0, :, :, 1:] = 0
        expected_dx[1, :, 3:, :] = 0
        expected_dx[2, :, :, 5:] = 0
        # Check expected output
        np.testing.assert_allclose(expected_dx, dx.data)


if __name__ == "__main__":
    unittest.main()
