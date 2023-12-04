import unittest

import torch

from laia.data import PaddedTensor
from laia.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d


class AdaptiveAvgPool2dTest(unittest.TestCase):
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
            dtype=torch.double,
            requires_grad=True,
        )

    def test_identity_tensor(self):
        h, w = 5, 6
        m = AdaptiveAvgPool2d(output_size=(h, w))
        x = torch.randn(2, 3, h, w, requires_grad=True)
        y = m(x)
        torch.testing.assert_close(y, x)
        (dx,) = torch.autograd.grad(y.sum(), inputs=(x,))
        torch.testing.assert_close(torch.ones(2, 3, h, w), dx)

    def test_forward_tensor(self):
        # N x C x H x W
        m = AdaptiveAvgPool2d(output_size=(1, 2))
        y = m(self.x)
        expected_y = torch.tensor(
            [
                # n = 0
                [
                    # c = 0
                    [[1 / 6, -9 / 6]],
                    # c = 1
                    [[18 / 6, 39 / 6]],
                ],
                # n = 1
                [
                    # c = 0
                    [[24 / 6, 16 / 6]],
                    # c = 1
                    [[20 / 6, 12 / 6]],
                ],
            ],
            dtype=self.x.dtype,
        )
        self.assertEqual(expected_y.size(), y.size())
        torch.testing.assert_close(y, expected_y)
        # Check against PyTorch's adaptive pooling
        y2 = torch.nn.functional.adaptive_avg_pool2d(self.x, output_size=(1, 2))
        torch.testing.assert_close(y, y2)

    def test_forward_padded_tensor(self):
        m = AdaptiveAvgPool2d(output_size=(1, 2))
        x = PaddedTensor(self.x, torch.tensor([[2, 2], [1, 3]]))
        y = m(x)
        expected_y = torch.tensor(
            [
                # n = 0
                [
                    # c = 0
                    [[3 / 2, 1 / 2]],
                    # c = 1
                    [[9 / 2, 6 / 2]],
                ],
                # n = 1
                [
                    # c = 0
                    [[12 / 2, 7 / 2]],
                    # c = 1
                    [[2 / 2, 6 / 2]],
                ],
            ],
            dtype=self.x.dtype,
        )
        torch.testing.assert_close(y, expected_y)

    def test_backward_tensor(self):
        m = AdaptiveAvgPool2d(output_size=(1, 2))
        torch.autograd.gradcheck(lambda x: m(x).sum(), self.x)

    def test_backward_padded_tensor(self):
        m = AdaptiveAvgPool2d(output_size=(1, 2))
        xs = torch.tensor([[2, 2], [1, 3]])
        torch.autograd.gradcheck(lambda x: m(PaddedTensor(x, xs)).sum(), self.x)


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
            dtype=torch.double,
            requires_grad=True,
        )

    def test_identity_tensor(self):
        h, w = 5, 6
        m = AdaptiveMaxPool2d(output_size=(h, w))
        x = torch.randn(2, 3, h, w, requires_grad=True)
        y = m(x)
        torch.testing.assert_close(y, x)
        (dx,) = torch.autograd.grad(y.sum(), inputs=x)
        torch.testing.assert_close(dx, torch.ones(2, 3, h, w))

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
            dtype=self.x.dtype,
        )
        self.assertEqual(expected_y.size(), y.size())
        torch.testing.assert_close(y, expected_y)
        # Check against PyTorch's adaptive pooling
        y2 = torch.nn.functional.adaptive_max_pool2d(self.x, output_size=(1, 2))
        torch.testing.assert_close(y, y2)

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
            dtype=self.x.dtype,
        )
        torch.testing.assert_close(y, expected_y)

    def test_backward_tensor(self):
        m = AdaptiveMaxPool2d(output_size=(1, 2))
        torch.autograd.gradcheck(lambda x: m(x).sum(), self.x)

    def test_backward_padded_tensor(self):
        m = AdaptiveMaxPool2d(output_size=(1, 2))
        xs = torch.tensor([[2, 2], [1, 3]])
        torch.autograd.gradcheck(lambda x: m(PaddedTensor(x, xs)).sum(), self.x)
