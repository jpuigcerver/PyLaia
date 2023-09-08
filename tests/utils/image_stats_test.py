import pytest
from PIL import Image

from laia.utils.image_stats import ImageStats

TR_TXT_TABLE = """
tmp-0 a a b b c
tmp-1 a a b b c
tmp-2 a a b b c
"""

VA_TXT_TABLE = """
tmp-3 a a b b c
tmp-4 a a b b c
tmp-6 a a b b c
"""

IMG_LIST = """
tmp-0
tmp-1
tmp-2
"""


def prepare_images(data_dir, sizes):
    for i, size in enumerate(sizes):
        im = Image.new(mode="L", size=size)
        im.save(str(data_dir / f"tmp-{i}.jpg"))


def prepare_training_data(data_dir, sizes):
    prepare_images(data_dir, sizes)
    tr_txt_table = data_dir / "tr.txt"
    tr_txt_table.write_text(TR_TXT_TABLE, "utf-8")
    va_txt_table = data_dir / "va.txt"
    va_txt_table.write_text(VA_TXT_TABLE, "utf-8")
    return [data_dir], str(tr_txt_table), str(va_txt_table)


def prepare_test_data(data_dir, sizes):
    prepare_images(data_dir, sizes)
    img_list = data_dir / "img_list.txt"
    img_list.write_text(IMG_LIST, "utf-8")
    return [data_dir], str(img_list)


@pytest.mark.parametrize(
    "sizes, expected_max_width, expected_is_fixed_height",
    [
        ([(556, 100)], 556, True),
        ([(556, 100), (600, 100), (1150, 100), (1200, 100)], 1200, True),
        ([(556, 100), (600, 110), (1150, 100), (1200, 100)], 1200, False),
    ],
)
def test_img_stats_fit_stage(
    tmpdir, sizes, expected_max_width, expected_is_fixed_height
):
    img_dirs, tr_txt_table, va_txt_table = prepare_training_data(tmpdir, sizes)
    img_stats = ImageStats(
        stage="fit",
        tr_txt_table=tr_txt_table,
        va_txt_table=va_txt_table,
        img_dirs=img_dirs,
    )
    assert img_stats.max_width == expected_max_width
    assert img_stats.is_fixed_height == expected_is_fixed_height


@pytest.mark.parametrize(
    "sizes, expected_max_width, expected_is_fixed_height",
    [
        ([(546, 100)], 546, True),
        ([(250, 100), (395, 100), (1150, 100)], 1150, True),
        ([(556, 128), (600, 110), (1200, 100)], 1200, False),
    ],
)
def test_img_stats_test_stage(
    tmpdir, sizes, expected_max_width, expected_is_fixed_height
):
    img_dirs, img_list = prepare_test_data(tmpdir, sizes)
    img_stats = ImageStats(
        stage="test",
        img_list=img_list,
        img_dirs=img_dirs,
    )
    assert img_stats.max_width == expected_max_width
    assert img_stats.is_fixed_height == expected_is_fixed_height
