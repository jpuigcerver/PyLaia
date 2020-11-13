import os
from pathlib import Path

import pytest

from laia.data import ImageDataset, TextImageFromTextTableDataset
from laia.data.text_image_from_text_table_dataset import (
    _get_images_and_texts_from_text_table,
    _load_text_table_from_file,
    find_image_filepath_from_id,
)


def test_text_image_from_text_table_dataset_empty():
    dataset = TextImageFromTextTableDataset([])
    assert len(dataset) == 0


def test_text_image_from_text_table_dataset(tmpdir, monkeypatch):
    monkeypatch.setattr(ImageDataset, "__getitem__", lambda *_: {"img": None})
    f = tmpdir / "foo"
    f.write(None)
    txt = "12 3 4"
    dataset = TextImageFromTextTableDataset([f"{f} {txt}"])
    assert len(dataset) == 1
    assert list(dataset[0].keys()) == ["img", "txt", "id"]
    assert dataset[0]["id"] == f
    assert dataset[0]["txt"] == txt


def test_find_image_filepath_from_id_not_found(tmpdir):
    filepath = find_image_filepath_from_id(
        "bar", tmpdir, img_extensions=[".jpg", ".png"]
    )
    assert filepath is None


@pytest.mark.parametrize("id_has_ext", [False, True])
def test_find_image_filepath_from_id(tmpdir, id_has_ext):
    img_id = "foo.PNG" if id_has_ext else "foo"
    img_dir = tmpdir.mkdir("dir")
    expected = img_dir / "foo.PNG"
    expected.write(None)
    filepath = find_image_filepath_from_id(
        img_id, img_dir, img_extensions=[".jpg", ".png"]
    )
    assert filepath == expected


def test_load_text_table_from_file(tmpdir, caplog):
    data = [" 1  2 3 4 ", " ", " # this is a test", "foo bar", "baz  "]
    f = tmpdir / "test.txt"
    f.write_text("\n".join(data), encoding="utf-8")
    test_cases = (str(f), Path(f), open(f), data)
    for case in test_cases:
        out = _load_text_table_from_file(case)
        assert list(out) == [("1", "2 3 4"), ("foo", "bar")]
    assert caplog.messages.count(
        "No text found for image ID 'baz', ignoring example..."
    ) == len(test_cases)


@pytest.mark.parametrize("img_list", [[], ["1 foo", "2 bar"]])
def test_get_images_and_texts_from_text_table_not_found(tmpdir, caplog, img_list):
    ids, filepaths, txts = _get_images_and_texts_from_text_table(img_list)
    assert ids == filepaths == txts == []
    assert sum(m.startswith("No image file found") for m in caplog.messages) == len(
        img_list
    )


def test_get_images_and_texts_from_text_table_with_dirs(tmpdir, caplog):
    table_file = ["1 foo", "2 bar", "3 baz"]
    # create test directories
    img_dirs = [tmpdir.mkdir(d) for d in ("dir1", "dir2")]
    expected = []
    for img, dir in zip(["1", "2"], img_dirs):
        filename = f"{img}.JPG"
        # create test images
        dir.join(filename).write(None)
        expected.append(dir / filename)
    ids, filepaths, txts = _get_images_and_texts_from_text_table(
        table_file, img_dirs=img_dirs
    )
    assert ids == ["1", "2"]
    assert filepaths == expected
    assert txts == ["foo", "bar"]
    assert (
        caplog.messages.count(
            "No image file found for image ID '3', ignoring example..."
        )
        == 1
    )


def test_get_img_ids_and_filepaths_without_dirs(tmpdir, caplog):
    img_names = ["1.jpeg", "2.png", "3.JPG"]
    img_list = [tmpdir / name for name in img_names]
    for img in img_list[:-1]:
        img.write(None)
    table_file = [f"{a} {b}" for a, b in zip(img_list, ["foo", "bar", "baz"])]
    ids, filepaths, txts = _get_images_and_texts_from_text_table(table_file)
    assert ids == filepaths == img_list[:-1]
    assert txts == ["foo", "bar"]
    assert (
        caplog.messages.count(
            f"No image file found for image ID '{img_list[-1]}', ignoring example..."
        )
        == 1
    )
