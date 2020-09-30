import pytest

from laia.data.image_from_list_dataset import (
    _get_img_ids_and_filepaths,
    _load_image_list_from_file,
)


@pytest.mark.parametrize("separator", ["\n", " \n "])
def test_load_image_list_from_file(tmpdir, separator):
    img_list = tmpdir / "test.lst"
    expected = [" # foo", "bar", "baz", "this is a test"]
    img_list.write_text(separator.join(expected), "utf-8")
    imgs = _load_image_list_from_file(str(img_list))
    assert imgs == expected[1:]


@pytest.mark.parametrize("img_list", [[], ["foo", "bar"]])
def test_get_ids_and_images_from_img_list_not_found(tmpdir, caplog, img_list):
    ids, filepaths = _get_img_ids_and_filepaths(img_list, img_dirs=[tmpdir])
    assert ids == [] == filepaths
    assert sum(m.startswith("No image file was found") for m in caplog.messages) == len(
        img_list
    )


def test_get_img_ids_and_filepaths_with_dirs(tmpdir, caplog):
    img_list = ["foo", "bar", "baz"]
    # create test directories
    img_dirs = [tmpdir.mkdir(d) for d in ("dir1", "dir2")]
    expected = []
    for img, dir in zip(img_list[:-1], img_dirs):
        filename = f"{img}.jpg"
        # create test images
        dir.join(filename).write(None)
        expected.append(str(dir / filename))
    ids, imgs = _get_img_ids_and_filepaths(img_list, img_dirs=img_dirs)
    assert ids == img_list[:-1]
    assert imgs == expected
    assert (
        caplog.messages.count(
            f'No image file was found for image ID "{img_list[-1]}", ignoring example...'
        )
        == 1
    )


def test_get_img_ids_and_filepaths_without_dirs(tmpdir, caplog):
    img_names = ["foo", "bar", "baz"]
    img_list = [tmpdir / f"{name}" for name in img_names]
    [img.write(None) for img in img_list[:-1]]
    ids, imgs = _get_img_ids_and_filepaths(map(str, img_list), img_dirs=None)
    assert ids == img_list[:-1] == imgs
    assert (
        caplog.messages.count(
            f'No image file was found for image ID "{img_list[-1]}", ignoring example...'
        )
        == 1
    )
