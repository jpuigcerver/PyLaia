import unittest

import torch

from laia.data import PaddedTensor
from laia.nn.temporal_pyramid_maxpool_2d import TemporalPyramidMaxPool2d

try:
    import nnutils_pytorch

    nnutils_installed = True
except ImportError:
    nnutils_installed = False


class TemporalPyramidMaxPool2dTest(unittest.TestCase):
    def _run_test_tensor(self, use_nnutils):
        x = torch.randn(3, 5, 7, 8, requires_grad=True)
        layer = TemporalPyramidMaxPool2d(levels=[1, 2], use_nnutils=use_nnutils)
        y = layer(x)
        self.assertEqual((3, 5 * (1 + 2)), y.size())
        (dx,) = torch.autograd.grad([torch.sum(y)], [x])

        # Check gradient w.r.t. x
        _, i1 = x.view(3, 5, 7 * 8).max(dim=2)
        _, i21 = x[:, :, :, :4].contiguous().view(3, 5, 7 * 4).max(dim=2)
        _, i22 = x[:, :, :, 4:].contiguous().view(3, 5, 7 * 4).max(dim=2)

        expected_dx = torch.zeros(3, 5, 7, 8)
        for n in range(3):
            for c in range(5):
                i, j = i1[n, c].item() // 8, i1[n, c].item() % 8
                expected_dx[n, c, i, j] += 1.0

                i, j = i21[n, c].item() // 4, i21[n, c].item() % 4
                expected_dx[n, c, i, j] += 1.0

                i, j = i22[n, c].item() // 4, i22[n, c].item() % 4
                expected_dx[n, c, i, j + 4] += 1.0

        torch.testing.assert_allclose(dx, expected_dx)

    @staticmethod
    def _run_test_padded_tensor(use_nnutils):
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
            requires_grad=True,
            dtype=torch.float,
        )
        layer = TemporalPyramidMaxPool2d(levels=[1, 2], use_nnutils=use_nnutils)
        y = layer(PaddedTensor(x, torch.tensor([[3, 4]])))
        (dx,) = torch.autograd.grad([torch.sum(y)], [x])

        # Expected gradient w.r.t. inputs
        expected_dx = torch.zeros(1, 1, 4, 8)
        expected_dx[0, 0, 2, 3] = 2
        expected_dx[0, 0, 2, 1] = 1

        # Check output and gradient w.r.t input
        torch.testing.assert_allclose(y, torch.tensor([[20.0, 18.0, 20.0]]))
        torch.testing.assert_allclose(dx, expected_dx)

    @unittest.skipIf(not nnutils_installed, "nnutils does not seem installed")
    def test_tensor_nnutils_backend(self):
        self._run_test_tensor(use_nnutils=True)

    def test_tensor_pytorch_backend(self):
        self._run_test_tensor(use_nnutils=False)

    @unittest.skipIf(not nnutils_installed, "nnutils does not seem installed")
    def test_padded_tensor_nnutils_backend(self):
        self._run_test_padded_tensor(use_nnutils=True)

    def test_padded_tensor_pytorch_backend(self):
        self._run_test_padded_tensor(use_nnutils=False)


if __name__ == "__main__":
    unittest.main()
