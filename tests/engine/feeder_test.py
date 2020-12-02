import pytest
import torch

from laia.engine import ImageFeeder, ItemFeeder


def test_item_feeder():
    feeder = ItemFeeder("foo")
    expected = "bar"
    x = {"foo": expected, "baz": 1}
    assert feeder(x) == expected


def test_item_feeder_raises():
    feeder = ItemFeeder("foo")
    with pytest.raises(AssertionError, match="Could not find key"):
        feeder({})


@pytest.mark.parametrize(
    ["x", "expected"],
    [
        (torch.empty(10, 20), (1, 1, 10, 20)),
        ((torch.tensor(1), torch.empty(1, 3)), (1, 1, 1, 1)),
    ],
)
def test_image_feeder(x, expected):
    feeder = ImageFeeder()
    assert feeder(x).data.size() == expected


def test_view_as_4d_raises():
    with pytest.raises(ValueError, match="Tensor with 5 dimensions"):
        ImageFeeder.view_as_4d(torch.empty(1, 1, 1, 1, 1))
