import pytest

from laia.data import ImageDataset, TextImageDataset


def test_text_image_dataset_empty():
    dataset = TextImageDataset([], [])
    assert len(dataset) == 0


def test_text_image_dataset_different_length():
    with pytest.raises(AssertionError):
        TextImageDataset([], ["foo"])


@pytest.mark.parametrize("transform", [None, lambda x: 1])
def test_image_dataset(transform):
    def monkeypatch(*_):
        return {"img": None}

    ImageDataset.__getitem__ = monkeypatch
    dataset = TextImageDataset(["foo.jpg"], ["bar"], txt_transform=transform)
    assert len(dataset) == 1
    assert list(dataset[0].keys()) == ["img", "txt"]
    assert dataset[0]["txt"] == "bar" if transform is None else 1
