import itertools
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

    def _run_test_forward(
        self, dtype, device, reduction, average_frames
    ):
        # Size: T x N x 3
        x = log_softmax(
            torch.tensor(
                [
                    [[0, 1, 2], [2, 3, 1], [0, 0, 1]],
                    [[-1, -1, 1], [-3, -2, 2], [1, 0, 0]],
                    [[0, 0, 0], [0, 0, 1], [1, 1, 1]],
                    [[0, 0, 2], [0, 0, -1], [0, 2, 1]],
                ],
                dtype=dtype,
                device=device,
            ),
            dim=-1,
        )
        y = [[1], [1, 1, 2, 1], [1, 2, 2]]  # targets
        paths0 = torch.tensor(
            [
                x[0, 0, a] + x[1, 0, b] + x[2, 0, c] + x[3, 0, d]
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
            dtype=dtype,
            device=device,
        )
        paths2 = x[0, 2, 1] + x[1, 2, 2] + x[2, 2, 0] + x[3, 2, 2]
        ctc = CTCLoss(
            reduction=reduction,
            average_frames=average_frames,
        )
        loss = ctc(x, y, batch_ids=["ID1", "ID2", "ID3"]).to(device)
        expected = torch.stack([-torch.logsumexp(paths0, dim=0), -paths2])
        if average_frames:
            expected = expected / 4.0
        if reduction == "sum":
            expected = torch.sum(expected)
        elif reduction == "mean":
            expected = torch.mean(expected)
        torch.testing.assert_allclose(expected.cpu(), loss.cpu())

    def _run_test_backward(
        self, dtype, device, reduction, average_frames
    ):
        ctc_logger = log.get_logger("laia.losses.ctc_loss")
        prev_level = ctc_logger.getEffectiveLevel()
        ctc_logger.setLevel(log.ERROR)
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
        ctc = CTCLoss(
            reduction=reduction,
            average_frames=average_frames,
        )
        gradcheck(lambda x: ctc(x, y), (x,))
        ctc_logger.setLevel(prev_level)


def _generate_test(
    test_name, method, dtype, device, reduction, average_frames
):
    avg_str = "avg_frames" if average_frames else "no_avg_frames"
    setattr(
        CTCLossTest,
        test_name
        + "_{}_{}_{}_{}".format(
            avg_str, reduction, device, str(dtype)[6:]
        ),
        lambda self: getattr(self, method)(
            implementation, dtype, device, reduction, average_frames
        ),
    )


dtypes = (torch.float, torch.double)
devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
average_frames = [False, True]
reductions = ["none", "mean", "sum"]
for dtype, device, avg_frames, reduction in itertools.product(
    dtypes, devices, average_frames, reductions
):
    _generate_test(
        "test_forward",
        "_run_test_forward",
        dtype,
        device,
        reduction,
        avg_frames,
    )
    # Check gradients only for double
    if dtype == torch.double:
        _generate_test(
            "test_backward",
            "_run_test_backward",
            dtype,
            device,
            reduction,
            avg_frames,
        )


if __name__ == "__main__":
    unittest.main()
