from __future__ import absolute_import
from __future__ import division

import unittest
import numpy as np

from laia.data import PaddedTensor
from laia.nn import TemporalPyramidMaxPool2d
import torch
from torch.autograd import Variable


class TemporalPyramidMaxPool2dTest(unittest.TestCase):

    def _test_tensor(self, use_nnutils):
        x = Variable(torch.randn(3, 5, 7, 8), requires_grad=True)
        layer = TemporalPyramidMaxPool2d(levels=2, use_nnutils=use_nnutils)
        y = layer(x)
        self.assertEqual((3, 5 * (1 + 2)), y.data.size())

        dx, = torch.autograd.grad([torch.sum(y)], [x])

        # Check gradient w.r.t. x
        _, i1 = x.data.view(3, 5, 7 * 8).max(dim=2)
        _, i21 = x.data[:, :, :, :4].contiguous().view(3, 5, 7 * 4).max(dim=2)
        _, i22 = x.data[:, :, :, 4:].contiguous().view(3, 5, 7 * 4).max(dim=2)

        expected_dx = torch.zeros(3, 5, 7, 8)
        for n in range(3):
            for c in range(5):
                i, j = i1[n, c] // 8, i1[n, c] % 8
                expected_dx[n, c, i, j] += 1.0

                i, j = i21[n, c] // 4, i21[n, c] % 4
                expected_dx[n, c, i, j] += 1.0

                i, j = i22[n, c] // 4, i22[n, c] % 4
                expected_dx[n, c, i, j + 4] += 1.0

        np.testing.assert_allclose(expected_dx, dx.data)

    def _test_padded_tensor(self, use_nnutils):
        x = Variable(
            torch.Tensor(
                [
                    [
                        [
                            [1, 2, 3, 4, 5, 6, 7, 8],
                            [9, 10, 11, 12, 13, 14, 15, 16],
                            [17, 18, 19, 20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29, 30, 31, 32],
                        ]
                    ]
                ]
            ),
            requires_grad=True,
        )
        xs = Variable(torch.LongTensor([[3, 4]]))
        layer = TemporalPyramidMaxPool2d(levels=2, use_nnutils=use_nnutils)
        y = layer(PaddedTensor(data=x, sizes=xs))
        dx, = torch.autograd.grad([torch.sum(y)], [x])

        # Expected gradient w.r.t. inputs
        expected_dx = torch.zeros(1, 1, 4, 8)
        expected_dx[0, 0, 2, 3] = 2
        expected_dx[0, 0, 2, 1] = 1

        # Check output and gradient w.r.t input
        np.testing.assert_allclose(torch.Tensor([[20, 18, 20]]), y.data)
        np.testing.assert_allclose(expected_dx, dx.data)

    def test_tensor_nnutils_backend(self):
        self._test_tensor(use_nnutils=True)

    def test_tensor_pytorch_backend(self):
        self._test_tensor(use_nnutils=False)

    def test_padded_tensor_nnutils_backend(self):
        self._test_padded_tensor(use_nnutils=True)

    def test_padded_tensor_pytorch_backend(self):
        self._test_padded_tensor(use_nnutils=False)


if __name__ == "__main__":
    unittest.main()
