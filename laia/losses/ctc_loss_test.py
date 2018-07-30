from __future__ import absolute_import

import unittest

import torch
from torch.autograd import gradcheck
from torch.nn.functional import log_softmax

import laia.common.logging as log
from laia.losses.ctc_loss import (
    copy_valid_indices,
    set_zeros_in_errors,
    get_valids_and_errors,
    CTCLoss,
)

log.basic_config()


class CTCLossTest(unittest.TestCase):
    def test_copy_valid_indices(self):
        acts = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        )
        labels, act_lens, valid_indices = [[1], [2], [3]], [1, 2, 3], [1]
        acts, target, act_lens = copy_valid_indices(
            acts, labels, act_lens, valid_indices
        )
        expected_acts = torch.tensor([[[4, 5, 6]], [[13, 14, 15]], [[22, 23, 24]]])
        expected_target, expected_act_lens = [[2]], [2]
        self.assertTrue(torch.equal(expected_acts, acts))
        self.assertEqual(expected_target, target)
        self.assertEqual(expected_act_lens, act_lens)

    def test_copy_valid_indices_when_empty(self):
        acts = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        )
        labels, act_lens, valid_indices = [[1], [2], [3]], [1, 2, 3], []
        output = copy_valid_indices(acts, labels, act_lens, valid_indices)
        expected = None, [], [], []
        self.assertEqual(expected, output)

    def test_set_zeros_in_errors(self):
        input = torch.tensor(
            [
                [[1, 2, 3], [7, 8, 9]],
                [[10, 11, 12], [16, 17, 18]],
                [[19, 20, 21], [25, 26, 27]],
            ],
            dtype=torch.float,
        )
        output = set_zeros_in_errors([3, 3, 3], input, [0, 2])
        expected = torch.tensor(
            [
                [[1, 2, 3], [0, 0, 0], [7, 8, 9]],
                [[10, 11, 12], [0, 0, 0], [16, 17, 18]],
                [[19, 20, 21], [0, 0, 0], [25, 26, 27]],
            ],
            dtype=torch.float,
        )
        torch.testing.assert_allclose(output, expected)

    def test_get_valids_and_errors(self):
        act_lens = [4, 4, 4, 5]
        labels = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 2, 3], [1, 2, 2, 3]]
        output = get_valids_and_errors(act_lens, labels)
        expected = [0, 3], [1, 2]
        self.assertEqual(expected, output)

    def test_get_valids_and_errors_when_wrong_size(self):
        act_lens = [1, 2]
        labels = [[1], [2], [3]]
        self.assertRaises(AssertionError, get_valids_and_errors, act_lens, labels)

    def _run_test_forward(self, dtype, device):
        # Size: T x N x 3
        x = torch.tensor(
            [
                [[0, 1, 2], [2, 3, 1], [0, 0, 1]],
                [[-1, -1, 1], [-3, -2, 2], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [1, 1, 1]],
                [[0, 0, 2], [0, 0, -1], [0, 2, 1]],
            ],
            dtype=dtype,
            device=device,
        )
        xn = log_softmax(x, dim=-1)
        paths0 = torch.tensor(
            [
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
            ],
            device=device,
        )
        paths2 = torch.tensor(xn[0, 2, 1] + xn[1, 2, 2] + xn[2, 2, 0] + xn[3, 2, 2])
        ctc = CTCLoss(size_average=False)
        y = [[1], [1, 1, 2, 1], [1, 2, 2]]
        loss = ctc(x, y, batch_ids=["ID1", "ID2", "ID3"]).to(device)
        expected = -torch.logsumexp(paths0 + paths2, 0)
        self.assertAlmostEqual(expected, loss)

    def _run_test_backward(self, dtype, device):
        # Size: T x N x 3
        x = torch.tensor(
            [
                [[0, 1, 2], [2, 3, 1], [0, 0, 1]],
                [[-1, -1, 1], [-3, -2, 2], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [1, 1, 1]],
                [[0, 0, 2], [0, 0, -1], [0, 2, 1]],
            ],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y = [[1], [1, 1, 2, 1], [1, 2, 2]]
        ctc = CTCLoss(size_average=False)
        self.assertTrue(gradcheck(lambda x, y: ctc(x, y), (x, y)))


def _generate_tests(dtype, test_name):
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        setattr(
            CTCLossTest,
            "test_forward_{}_{}".format(device, test_name),
            lambda self: self._run_test_forward(dtype, device),
        )
        setattr(
            CTCLossTest,
            "test_backward_{}_{}".format(device, test_name),
            lambda self: self._run_test_backward(dtype, device),
        )


for dtype in (torch.float,):
    _generate_tests(dtype, str(dtype)[6:])

if __name__ == "__main__":
    unittest.main()
