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


def prepare_data(tmpdir, sizes):
    for i, size in enumerate(sizes):
        im = Image.new(mode="L", size=size)
        im.save(str(tmpdir / f"tmp-{i}.jpg"))
    tr_txt_table = tmpdir / "tr.txt"
    tr_txt_table.write_text(TR_TXT_TABLE, "utf-8")
    va_txt_table = tmpdir / "va.txt"
    va_txt_table.write_text(VA_TXT_TABLE, "utf-8")
    return [tmpdir], str(tr_txt_table), str(va_txt_table)


@pytest.mark.parametrize(
    "sizes, expected_max_width, expected_is_fixed_height",
    [
        ([(556, 100)], 556, True),
        ([(556, 100), (600, 100), (1150, 100), (1200, 100)], 1200, True),
        ([(556, 100), (600, 110), (1150, 100), (1200, 100)], 1200, False),
    ],
)
def test_img_stats(tmpdir, sizes, expected_max_width, expected_is_fixed_height):
    img_dirs, tr_txt_table, va_txt_table = prepare_data(tmpdir, sizes)
    img_stats = ImageStats(
        img_dirs,
        tr_txt_table,
        va_txt_table,
    )
    assert img_stats.max_width == expected_max_width
    assert img_stats.is_fixed_height == expected_is_fixed_height
