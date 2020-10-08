import os
from unittest.mock import MagicMock

import numpy as np

from laia.dummies import DummyMNISTLines


def test_get_indices_without_spaces():
    expected = [5, 10, 15, 20, 25, 30, 35, 40]
    np.random.randint = MagicMock(return_value=len(expected))
    np.random.choice = MagicMock(return_value=expected)
    assert DummyMNISTLines.get_indices(10, 0) == expected


def test_get_indices_with_spaces():
    choices1 = [5, 10, 15, 20, 25, 30, 35, 40]
    choices2 = [1, 5, 6, 7]
    np.random.randint = MagicMock(return_value=len(choices1))
    np.random.choice = MagicMock(side_effect=[choices1, choices2])
    out = DummyMNISTLines.get_indices(10, 0, samples_per_space=3)
    assert out == [5, "sp", 10, 15, 20, 25, "sp", 30, "sp", 35, "sp", 40]


def test_concatenate():
    h, w = 10, 15
    dataset = [
        (np.full((h, w), 11), "a"),
        (np.full((h, w), 12), "b"),
        (np.full((h, w), 13), "foo"),
        (np.full((h, w), 14), "bar"),
    ]
    indices = [0, 3, 1, "sp", 2]
    img, txt, mask = DummyMNISTLines.concatenate(
        dataset, h, w, indices, space_sym="test"
    )

    # check image values
    assert np.all(img[:, :w] == 11)
    assert np.all(img[:, w : w * 2] == 14)
    assert np.all(img[:, w * 2 : w * 3] == 12)
    assert np.all(img[:, w * 3 : w * 4] == 0)
    assert np.all(img[:, w * 4 :] == 13)

    # check label text
    assert txt == "a bar b test foo"

    # check mask values
    assert np.all(mask[:, :w] == 0)
    assert np.all(mask[:, w : w * 2] == 0)
    assert np.all(mask[:, w * 2 : w * 3] == 0)
    assert np.all(mask[:, w * 3 : w * 4] == 1)
    assert np.all(mask[:, w * 4 :] == 0)


def test_prepare_data(tmpdir):
    data_module = DummyMNISTLines(max_length=5, tr_n=5, va_n=3)
    indices = [5, "sp", "sp", 10, 25, "sp", 30]
    expected_labels = {
        "tr": "2 <space> <space> 3 2 <space> 3",
        "va": "1 <space> <space> 0 0 <space> 3",
    }
    data_module.get_indices = MagicMock(return_value=indices)
    data_module.prepare_data()

    for partition in ("tr", "va"):
        # check generated images
        image_ids = [f"{partition}-{i}" for i in range(data_module.n[partition])]
        assert set(os.listdir(data_module.root / partition)) == {
            img + ".jpg" for img in image_ids
        }

        # check generated ground-truth file
        gt_file = data_module.root / f"{partition}.gt"
        assert gt_file.exists()
        lines = [l.strip() for l in gt_file.read_text().split("\n") if l]
        for i, l in enumerate(lines):
            img_id, labels = l.split(maxsplit=1)
            assert img_id == image_ids[i]
            assert labels == expected_labels[partition]

        # check generated indices file
        indices_file = data_module.root / f"{partition}.indices"
        assert indices_file.exists()
        lines = [l.strip() for l in indices_file.read_text().split("\n") if l]
        for i, l in enumerate(lines):
            img_id, indices_str = l.split(maxsplit=1)
            assert img_id == image_ids[i]
            assert indices_str == repr([str(x) for x in indices])
