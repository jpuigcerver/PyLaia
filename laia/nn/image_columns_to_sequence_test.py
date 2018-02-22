from __future__ import absolute_import

import unittest

import numpy as np
import torch
from torch.autograd import Variable

from laia.data import PaddedTensor
from laia.nn import ImageColumnsToSequence


class ImageColumnsToSequenceTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(torch.FloatTensor([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9],
                                        [10, 11, 12]]))
        m = ImageColumnsToSequence(rows=4)
        y = m(x)
        np.testing.assert_allclose(y.data, np.array([[[1, 4, 7, 10]],
                                                     [[2, 5, 8, 11]],
                                                     [[3, 6, 9, 12]]]))

    def test_backward(self):
        x = Variable(torch.FloatTensor([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9],
                                        [10, 11, 12]]),
                     requires_grad=True)
        m = ImageColumnsToSequence(rows=4)
        y = m(x)
        dx, = torch.autograd.grad([y.sum()], [x])
        np.testing.assert_allclose(dx.data, np.array([[1, 1, 1],
                                                      [1, 1, 1],
                                                      [1, 1, 1],
                                                      [1, 1, 1]]))

    def test_forward_with_size(self):
        x = Variable(torch.FloatTensor([[[[1, 2, 3],
                                          [4, 5, 6]]],
                                        [[[7, 8, 0],
                                          [10, 11, 0]]]]))
        xs = torch.LongTensor([[2, 3], [2, 2]])
        m = ImageColumnsToSequence(rows=2)
        y, ys = m(PaddedTensor(x, xs))
        np.testing.assert_allclose(y.data, np.array([[[1, 4],
                                                      [7, 10]],
                                                     [[2, 5],
                                                      [8, 11]],
                                                     [[3, 6],
                                                      [0, 0]]]))
        self.assertEqual(ys, [3, 2])

    def test_backward_with_size(self):
        x = Variable(torch.FloatTensor([[[[1, 2, 3],
                                          [4, 5, 6]]],
                                        [[[7, 8, 0],
                                          [10, 11, 0]]]]),
                     requires_grad=True)
        xs = torch.LongTensor([[2, 3], [2, 2]])
        m = ImageColumnsToSequence(rows=2)
        y, ys = m(PaddedTensor(x, xs))
        dx, = torch.autograd.grad(
            [y[0, :, :].sum() + y[1, :, :].sum() + y[2, 0, :].sum()],
            [x])
        np.testing.assert_allclose(dx.data, np.array([[[[1, 1, 1],
                                                        [1, 1, 1]]],
                                                      [[[1, 1, 0],
                                                        [1, 1, 0]]]]))

    def test_forward_backward_packed(self):
        x = Variable(torch.FloatTensor([[[[1, 2, 3],
                                          [4, 5, 6]]],
                                        [[[7, 8, 0],
                                          [10, 11, 0]]]]), requires_grad=True)
        xs = torch.LongTensor([[2, 3], [2, 2]])
        m = ImageColumnsToSequence(rows=2, return_packed=True)
        # Test forward
        y = m(PaddedTensor(x, xs))
        np.testing.assert_allclose(y.data.data, np.array([[1, 4],
                                                          [7, 10],
                                                          [2, 5],
                                                          [8, 11],
                                                          [3, 6]]))
        self.assertEqual(y.batch_sizes, [2, 2, 1])
        # Test backward pass
        dx, = torch.autograd.grad([y.data.sum()], [x])
        np.testing.assert_allclose(dx.data, np.array([[[[1, 1, 1],
                                                        [1, 1, 1]]],
                                                      [[[1, 1, 0],
                                                        [1, 1, 0]]]]))


if __name__ == '__main__':
    unittest.main()
