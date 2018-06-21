from __future__ import absolute_import
from __future__ import division

import unittest

import numpy as np
import torch
from laia.data import PaddedTensor
from laia.nn.adaptive_avgpool_2d import AdaptiveAvgPool2d
from torch.autograd import Variable, gradcheck
from torch.nn.functional import adaptive_avg_pool2d


class AdaptiveAvgPool2dTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.Tensor(
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
            ]
        )

    @staticmethod
    def test_identity_tensor():
        h, w = 5, 6
        m = AdaptiveAvgPool2d(output_size=(h, w))
        x = Variable(torch.randn(2, 3, h, w), requires_grad=True)
        y = m(x)
        np.testing.assert_almost_equal(y.data.cpu().numpy(), x.data.cpu().numpy())
        dx, = torch.autograd.grad(y.sum(), inputs=(x,))
        np.testing.assert_almost_equal(dx.data.cpu().numpy(), np.ones((2, 3, h, w)))

    def test_forward_tensor(self):
        # N x C x H x W
        m = AdaptiveAvgPool2d(output_size=(1, 2))
        y = m(self.x)
        expected_y = np.asarray(
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
            ]
        )
        self.assertListEqual(list(y.size()), list(expected_y.shape))
        np.testing.assert_almost_equal(y.data.cpu().numpy(), expected_y)
        # Check against PyTorch's adaptive pooling
        y2 = adaptive_avg_pool2d(self.x, output_size=(1, 2))
        np.testing.assert_almost_equal(y.data.cpu().numpy(), y2.data.cpu().numpy())

    def test_forward_padded_tensor(self):
        m = AdaptiveAvgPool2d(output_size=(1, 2))
        x = PaddedTensor(
            data=self.x, sizes=Variable(torch.LongTensor([[2, 2], [1, 3]]))
        )
        y = m(x)
        expected_y = np.asarray(
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
            ]
        )
        np.testing.assert_almost_equal(y.data.cpu().numpy(), expected_y)

    def test_backward_tensor(self):
        m = AdaptiveAvgPool2d(output_size=(1, 2))

        def wrap_func(x):
            return m(x).sum()

        gradcheck(func=wrap_func, inputs=(self.x,))

    def test_backward_padded_tensor(self):
        m = AdaptiveAvgPool2d(output_size=(1, 2))

        def wrap_func(x):
            return m(
                PaddedTensor(data=x, sizes=Variable(torch.LongTensor([[2, 2], [1, 3]])))
            ).sum()

        gradcheck(func=wrap_func, inputs=(self.x,))
