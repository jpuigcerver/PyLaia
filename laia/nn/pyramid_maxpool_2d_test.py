from __future__ import absolute_import

import unittest

import numpy as np
import torch
from torch.autograd import Variable

from laia.data import PaddedTensor

try:
    from laia.nn.pyramid_maxpool_2d import PyramidMaxPool2d
except ImportError:
    PyramidMaxPool2d = None


@unittest.skipIf(PyramidMaxPool2d is None, "nnutils does not seem installed")
class PyramidMaxPool2dTest(unittest.TestCase):
    def _run_test_tensor(self, use_nnutils):
        x = torch.randn(3, 5, 7, 8)
        layer = PyramidMaxPool2d(levels=[1, 2], use_nnutils=use_nnutils)
        y = layer(Variable(x))
        self.assertEqual((3, 5 * (1 + 2 * 2)), y.data.size())

        # Check gradient
        def wrap_func(xx):
            return torch.sum(layer(xx))

        torch.autograd.gradcheck(wrap_func, inputs=(x,))

    @staticmethod
    def _run_test_padded_tensor(use_nnutils):
        x = torch.Tensor(
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
        )
        xs = torch.LongTensor([[3, 4]])
        layer = PyramidMaxPool2d(levels=[1, 2], use_nnutils=use_nnutils)
        y = layer(PaddedTensor(data=Variable(x), sizes=Variable(xs)))
        np.testing.assert_allclose(torch.Tensor([[20, 10, 12, 18, 20]]), y.data)

        # Check gradient
        def wrap_func(xx):
            return layer(PaddedTensor(data=xx, sizes=Variable(xs)))

        torch.autograd.gradcheck(wrap_func, inputs=(x,))

    def test_tensor_nnutils_backend(self):
        self._run_test_tensor(use_nnutils=True)

    def test_tensor_pytorch_backend(self):
        self._run_test_tensor(use_nnutils=False)

    def test_padded_tensor_nnutils_backend(self):
        self._run_test_padded_tensor(use_nnutils=True)

    def test_padded_tensor_pytorch_backend(self):
        self._run_test_padded_tensor(use_nnutils=False)


if __name__ == "__main__":
    unittest.main()
