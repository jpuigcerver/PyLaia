from __future__ import absolute_import

import unittest

import torch
from scipy.misc import logsumexp
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import numpy as np

from laia.losses.ctc_loss import (
    copy_valid_indices,
    set_zeros_in_errors,
    get_valids_and_errors,
    CTCLoss,
)


class HookTest(unittest.TestCase):

    def test_copy_valid_indices(self):
        acts = torch.Tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        )
        labels = [1, 2, 3]
        act_lens = [1, 2, 3]
        label_lens = [1, 2, 3]
        valid_indices = [1]
        output = copy_valid_indices(acts, labels, act_lens, label_lens, valid_indices)
        expected = [[[4, 5, 6]], [[13, 14, 15]], [[22, 23, 24]]], [2], [2], [2]
        np.testing.assert_equal(output[0].numpy(), expected[0])
        self.assertEqual(output[1:], expected[1:])

    def test_copy_valid_indices_when_empty(self):
        acts = torch.Tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        )
        labels = [1, 2, 3]
        act_lens = [1, 2, 3]
        label_lens = [1, 2, 3]
        valid_indices = []
        output = copy_valid_indices(acts, labels, act_lens, label_lens, valid_indices)
        expected = None, [], [], []
        self.assertEqual(output, expected)

    def test_set_zeros_in_errors(self):
        input = torch.Tensor(
            [
                [[1, 2, 3], [7, 8, 9]],
                [[10, 11, 12], [16, 17, 18]],
                [[19, 20, 21], [25, 26, 27]],
            ]
        )
        valid_indices = [0, 2]
        output = set_zeros_in_errors([3, 3, 3], input, valid_indices)
        expected = torch.Tensor(
            [
                [[1, 2, 3], [0, 0, 0], [7, 8, 9]],
                [[10, 11, 12], [0, 0, 0], [16, 17, 18]],
                [[19, 20, 21], [0, 0, 0], [25, 26, 27]],
            ]
        )
        np.testing.assert_equal(output.numpy(), expected.numpy())

    def test_get_valids_and_errors(self):
        act_lens = [309, 111, 191, 171, 59]
        label_lens = [63, 55, 46, 44, 74]
        output = get_valids_and_errors(act_lens, label_lens)
        expected = [0, 2, 3], [1, 4]
        self.assertEqual(output, expected)

    def test_get_valids_and_errors_when_wrong_size(self):
        act_lens = [1, 2]
        label_lens = [1, 2, 3]
        self.assertRaises(AssertionError, get_valids_and_errors, act_lens, label_lens)

    def test_forward(self):
        ctc = CTCLoss(size_average=False, length_average=False)
        # Size: T x N x 2
        x = Variable(
            torch.Tensor(
                [
                    [[0, 1], [2, 3]],
                    [[-1, -2], [-3, -4]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                ]
            )
        )
        y = [[1], [1, 2]]
        xn = log_softmax(x, dim=-1).data
        paths = [
            xn[0, 0, a] + xn[1, 0, b] + xn[2, 0, c] + xn[3, 0, d]
            for a, b, c, d in [
                (1, 1, 1, 1),
                (1, 1, 1, 0),
                (0, 1, 1, 1),
                (1, 1, 0, 0),
                (0, 1, 1, 0),
                (0, 0, 1, 1),
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ]
        ]
        loss, errors = ctc(x, y)
        self.assertAlmostEqual(loss.data[0], -logsumexp(paths))
        self.assertEqual(errors, [1])

    def test_backward(self):
        # TODO(jpuigcerver)
        pass


if __name__ == "__main__":
    unittest.main()
