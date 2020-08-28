import unittest

import torch

from laia.data import PaddedTensor
from laia.nn import ImageToSequence


class ImageToSequenceTest(unittest.TestCase):
    def test_forward(self):
        x = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float
        )
        m = ImageToSequence(columnwise=True)
        y = m(x)
        expected_y = torch.tensor(
            [[[1, 4, 7, 10]], [[2, 5, 8, 11]], [[3, 6, 9, 12]]], dtype=torch.float
        )
        torch.testing.assert_allclose(y, expected_y)

    def test_backward(self):
        x = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=torch.float,
            requires_grad=True,
        )
        m = ImageToSequence(columnwise=True)
        y = m(x)
        (dx,) = torch.autograd.grad([y.sum()], [x])
        torch.testing.assert_allclose(dx, torch.ones(4, 3))

    def test_forward_with_size(self):
        x = torch.tensor(
            [[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 0], [10, 11, 0]]]], dtype=torch.float
        )
        xs = torch.tensor([[2, 3], [2, 2]])
        m = ImageToSequence(columnwise=True)
        y, ys = m(PaddedTensor(x, xs))
        expected_y = torch.tensor(
            [[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [0, 0]]], dtype=torch.float
        )
        torch.testing.assert_allclose(y, expected_y)
        self.assertEqual(ys, [3, 2])

    def test_backward_with_size(self):
        x = torch.tensor(
            [[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 0], [10, 11, 0]]]],
            dtype=torch.float,
            requires_grad=True,
        )
        xs = torch.tensor([[2, 3], [2, 2]])
        m = ImageToSequence(columnwise=True)
        y, ys = m(PaddedTensor(x, xs))
        (dx,) = torch.autograd.grad(
            [y[0, :, :].sum() + y[1, :, :].sum() + y[2, 0, :].sum()], [x]
        )
        expected_dx = torch.tensor(
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 0], [1, 1, 0]]]], dtype=torch.float
        )
        torch.testing.assert_allclose(dx, expected_dx)

    def test_forward_backward_packed(self):
        x = torch.tensor(
            [[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 0], [10, 11, 0]]]],
            dtype=torch.float,
            requires_grad=True,
        )
        xs = torch.tensor([[2, 3], [2, 2]])
        m = ImageToSequence(columnwise=True, return_packed=True)
        # Test forward
        y = m(PaddedTensor(x, xs))
        expected_y = torch.tensor(
            [[1, 4], [7, 10], [2, 5], [8, 11], [3, 6]], dtype=torch.float
        )
        torch.testing.assert_allclose(y.data, expected_y)
        self.assertTrue(torch.equal(torch.tensor([2, 2, 1]), y.batch_sizes))
        # Test backward pass
        (dx,) = torch.autograd.grad([y.data.sum()], [x])
        expected_dx = torch.tensor(
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 0], [1, 1, 0]]]], dtype=torch.float
        )
        torch.testing.assert_allclose(dx, expected_dx)


if __name__ == "__main__":
    unittest.main()
