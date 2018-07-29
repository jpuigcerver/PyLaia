
from __future__ import absolute_import

import unittest

import torch
from torch.autograd import gradcheck

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
        y = layer(x)
        self.assertEqual((3, 5 * (1 + 2 * 2)), y.size())
        gradcheck(lambda x: torch.sum(layer(x)), inputs=x)

    def _run_test_padded_tensor(self, use_nnutils):
        x = torch.tensor(
            [
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [9, 10, 11, 12, 13, 14, 15, 16],
                        [17, 18, 19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30, 31, 32],
                    ]
                ]
            ],
            dtype=torch.float,
        )
        xs = torch.tensor([[3, 4]])
        layer = PyramidMaxPool2d(levels=[1, 2], use_nnutils=use_nnutils)
        y = layer(PaddedTensor(x, xs))
        torch.testing.assert_allclose(
            y, torch.tensor([[20, 10, 12, 18, 20]], dtype=torch.float)
        )
        gradcheck(lambda x: layer(PaddedTensor(x, xs)), inputs=x)

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
