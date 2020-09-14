import itertools
import unittest

import torch
from torch.autograd import gradcheck
from torch.nn.functional import log_softmax

import laia.common.logging as log
from laia.losses.ctc_loss import CTCLoss, get_valids_and_errors

log.config()


class CTCLossTest(unittest.TestCase):
    def test_get_valids_and_errors(self):
        xs = [4, 4, 4, 5]
        y = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 2, 3], [1, 2, 2, 3]]
        output = get_valids_and_errors(xs, y)
        expected = [0, 3], [1, 2]
        self.assertEqual(expected, output)

    def test_get_valids_and_errors_when_wrong_size(self):
        xs = [1, 2]
        y = [[1], [2], [3]]
        self.assertRaises(AssertionError, get_valids_and_errors, xs, y)

    def _run_test_forward(self, dtype, device, reduction, average_frames):
        # Size: T=4 x B=3 x C=3
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
        ctc = CTCLoss(reduction=reduction, average_frames=average_frames)
        loss = ctc(x, y, batch_ids=["ID1", "ID2", "ID3"]).to(device)
        expected = torch.stack([-torch.logsumexp(paths0, dim=0), -paths2])
        if average_frames:
            expected = expected / 4.0
        if reduction == "sum":
            expected = torch.sum(expected)
        elif reduction == "mean":
            expected = torch.mean(expected)
        torch.testing.assert_allclose(expected.cpu(), loss.cpu())

    def _run_test_backward(self, dtype, device, reduction, average_frames):
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
        ctc = CTCLoss(reduction=reduction, average_frames=average_frames)
        gradcheck(lambda x: ctc(x, y), (x,))
        ctc_logger.setLevel(prev_level)


def _generate_test(test_name, method, dtype, device, reduction, average_frames):
    avg_str = "avg_frames" if average_frames else "no_avg_frames"
    setattr(
        CTCLossTest,
        f"{test_name}_{avg_str}_{reduction}_{device}_{str(dtype)[6:]}",
        lambda self: getattr(self, method)(dtype, device, reduction, average_frames),
    )


dtypes = (torch.float, torch.double)
devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
average_frames = [False, True]
reductions = ["none", "mean", "sum"]
for dtype, device, avg_frames, reduction in itertools.product(
    dtypes, devices, average_frames, reductions
):
    _generate_test(
        "test_forward", "_run_test_forward", dtype, device, reduction, avg_frames
    )
    # Check gradients only for double
    if dtype == torch.double:
        _generate_test(
            "test_backward", "_run_test_backward", dtype, device, reduction, avg_frames
        )


if __name__ == "__main__":
    unittest.main()
