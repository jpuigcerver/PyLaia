import pytest
import torch
from torch.nn.functional import log_softmax

from laia.losses.ctc_loss import CTCLoss, get_valids_and_errors, transform_batch


def test_transform_batch():
    with pytest.raises(
        NotImplementedError, match=r"Not implemented for type <class 'NoneType'>"
    ):
        transform_batch(None)

    batch = torch.empty((3, 4, 5))
    x, xs = transform_batch(batch)
    torch.testing.assert_allclose(x, batch)
    assert xs == [3, 3, 3, 3]

    x = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]])
    xs = [3, 2, 1]
    batch = torch.nn.utils.rnn.pack_padded_sequence(x, xs)
    x_out, xs_out = transform_batch(batch)
    torch.testing.assert_allclose(x_out, x)
    assert xs_out.tolist() == xs


def test_get_valids_and_errors():
    xs = [4, 4, 4, 5]
    y = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 2, 3], [1, 2, 2, 3]]
    output = get_valids_and_errors(xs, y)
    expected = [0, 3], [1, 2]
    assert output == expected


def test_get_valids_and_errors_when_wrong_size():
    xs = [1, 2]
    y = [[1], [2], [3]]
    with pytest.raises(AssertionError):
        get_valids_and_errors(xs, y)


def test_ctc_loss_invalid_inputs():
    with pytest.raises(AssertionError, match=r"Blank index must be >= 0"):
        CTCLoss(blank=-1)
    with pytest.raises(AssertionError, match=r"Reduction foo is not supported"):
        CTCLoss(reduction="foo")
    with pytest.raises(AssertionError, match=r"Batch size does not match"):
        loss = CTCLoss()
        loss(torch.empty((4, 5, 3)), [[]])


def test_forward_all_ignored(caplog):
    loss = CTCLoss()
    loss(torch.empty((1, 2, 1)), [[1, 1], [1, 1, 1]])
    for m in (
        "The following samples in the batch were ignored for the loss computation: [0, 1]",
        "All samples in the batch were ignored!",
    ):
        assert caplog.messages.count(m) == 1


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("average_frames", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_forward(caplog, dtype, device, average_frames, reduction):
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
    assert (
        caplog.messages.count(
            "The following samples in the batch were ignored for the loss computation: ['ID2']"
        )
        == 1
    )


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("average_frames", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_backward(device, average_frames, reduction):
    # Size: T x N x 3
    x = torch.tensor(
        [
            [[0, 1, 2], [2, 3, 1], [0, 0, 1]],
            [[-1, -1, 1], [-3, -2, 2], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 1, 1]],
            [[0, 0, 2], [0, 0, -1], [0, 2, 1]],
        ],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    y = [[1], [1, 1, 2, 1], [1, 2, 2]]
    ctc = CTCLoss(reduction=reduction, average_frames=average_frames)
    torch.autograd.gradcheck(lambda x: ctc(x, y), x)
