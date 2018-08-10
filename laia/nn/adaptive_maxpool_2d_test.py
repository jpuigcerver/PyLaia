import unittest

import torch
from torch.autograd import gradcheck
from torch.nn.functional import adaptive_max_pool2d

from laia.data import PaddedTensor

try:
    from laia.nn.adaptive_maxpool_2d import AdaptiveMaxPool2d
except ImportError:
    AdaptiveMaxPool2d = None


@unittest.skipIf(AdaptiveMaxPool2d is None, "nnutils does not seem installed")
class AdaptiveMaxPool2dTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [
                # n = 0
                [
                    # c = 0
                    [
                        # y = 0
                        [1, 0, -1, 0],
                        # y = 1
                        [2, 1, 1, -2],
                        # y = 2
                        [-1, -2, -3, -4],
                    ],
                    # c = 1
                    [
                        # y = 0
                        [3, 1, 1, 7],
                        # y = 1
                        [6, 5, 21, 3],
                        # y = 2
                        [1, 2, 3, 4],
                    ],
                ],
                # n = 1
                [
                    # c = 0
                    [
                        # y = 0
                        [4, 8, -1, 5],
                        # y = 1
                        [-5, 6, 3, -2],
                        # y = 2
                        [5, 6, 5, 6],
                    ],
                    # c = 1
                    [
                        # y = 0
                        [-2, 4, 2, 1],
                        # y = 1
                        [2, 5, -3, 3],
                        # y = 2
                        [9, 2, 6, 3],
                    ],
                ],
            ],
            dtype=torch.float,
        )

    def test_identity_tensor(self):
        h, w = 5, 6
        m = AdaptiveMaxPool2d(output_size=(h, w))
        x = torch.randn(2, 3, h, w, requires_grad=True)
        y = m(x)
        torch.testing.assert_allclose(y, x)
        dx, = torch.autograd.grad(y.sum(), inputs=x)
        torch.testing.assert_allclose(dx, torch.ones(2, 3, h, w))

    def test_forward_tensor(self):
        m = AdaptiveMaxPool2d(output_size=(1, 2))
        y = m(self.x)
        expected_y = torch.tensor(
            [
                # n = 0
                [
                    # c = 0
                    [[2, 1]],
                    # c = 1
                    [[6, 21]],
                ],
                # n = 1
                [
                    # c = 0
                    [[8, 6]],
                    # c = 1
                    [[9, 6]],
                ],
            ],
            dtype=torch.float,
        )
        self.assertEqual(expected_y.size(), y.size())
        torch.testing.assert_allclose(y, expected_y)
        # Check against PyTorch's adaptive pooling
        y2 = adaptive_max_pool2d(self.x, output_size=(1, 2))
        torch.testing.assert_allclose(y, y2)

    def test_forward_padded_tensor(self):
        m = AdaptiveMaxPool2d(output_size=(1, 2))
        x = PaddedTensor(self.x, torch.tensor([[2, 2], [1, 3]]))
        y = m(x)
        expected_y = torch.tensor(
            [
                # n = 0
                [
                    # c = 0
                    [[2, 1]],
                    # c = 1
                    [[6, 5]],
                ],
                # n = 1
                [
                    # c = 0
                    [[8, 8]],
                    # c = 1
                    [[4, 4]],
                ],
            ],
            dtype=torch.float,
        )
        torch.testing.assert_allclose(y, expected_y)

    def test_backward_tensor(self):
        m = AdaptiveMaxPool2d(output_size=(1, 2))
        gradcheck(lambda x: m(x).sum(), self.x)

    def test_backward_padded_tensor(self):
        m = AdaptiveMaxPool2d(output_size=(1, 2))
        xs = torch.tensor([[2, 2], [1, 3]])
        gradcheck(lambda x: m(PaddedTensor(x, xs)).sum(), self.x)
