from __future__ import absolute_import

import unittest

import torch
from torch.autograd import Variable

from laia.data import PaddedTensor
from laia.nn.adaptive_pool_2d_base import AdaptivePool2dBase


class DummyPool2d(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, batch_input, output_sizes, batch_sizes):
        ctx.save_for_backward(batch_sizes)

        if output_sizes[0] is None:
            return batch_input[:, :, :, 0 : output_sizes[1]]
        elif output_sizes[1] is None:
            return batch_input[:, :, 0 : output_sizes[0], :]
        else:
            return batch_input[:, :, 0 : output_sizes[0], 0 : output_sizes[1]]

    @classmethod
    def backward(cls, ctx, grad_outputs):
        grad_inputs = torch.FloatTensor()


class AdaptivePool2dBaseTest(unittest.TestCase):
    def test_simple(self):
        x = Variable(
            torch.FloatTensor(
                [
                    [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [-9, -9, -9, -9, -9]]],
                    [
                        [
                            [11, 12, 13, -9, -9],
                            [14, 15, 16, -9, -9],
                            [17, 18, 19, -9, -9],
                        ]
                    ],
                ]
            ),
            requires_grad=True,
        )
        # TODO(jpuigcerver): Complete test


if __name__ == "__main__":
    unittest.main()
