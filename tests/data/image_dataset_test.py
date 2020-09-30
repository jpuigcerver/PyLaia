import numpy as np
import PIL
import pytest

from laia.data import ImageDataset


def test_image_dataset_empty():
    dataset = ImageDataset([])
    assert len(dataset) == 0


@pytest.mark.parametrize("transform", [None, lambda x: 1])
def test_image_dataset(monkeypatch, transform):
    expected_image = np.array([[1, 2], [3, 4]])
    monkeypatch.setattr(PIL.Image, "open", lambda _: expected_image)
    dataset = ImageDataset(["foo.jpg"], transform=transform)
    assert len(dataset) == 1
    assert list(dataset[0].keys()) == ["img"]
    if transform is None:
        np.testing.assert_allclose(expected_image, dataset[0]["img"])
    else:
        assert dataset[0]["img"] == 1
