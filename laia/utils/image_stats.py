#!/usr/bin/env python3
from functools import cached_property
from pathlib import Path
from typing import List, TextIO, Union

import imagesize

from laia.data.text_image_from_text_table_dataset import (
    _get_images_and_texts_from_text_table,
)


class ImageStats:
    """
    Compute statistics on the dataset

    Args:
        img_dir: Path to images
        tr_txt_table: Path to the train text table
        va_txt_table: Path to the validation text table
    """

    def __init__(
        self,
        tr_txt_table: Union[TextIO, str, List[str]],
        va_txt_table: Union[TextIO, str, List[str]],
        img_dirs: Optional[Union[List[str], str, List[Path], Path]] = None,
    ):
        self.tr_image_paths = _get_images_and_texts_from_text_table(
            tr_txt_table, img_dirs
        )[1]
        self.va_image_paths = _get_images_and_texts_from_text_table(
            va_txt_table, img_dirs
        )[1]
        sizes = list(map(imagesize.get, self.tr_image_paths + self.va_image_paths))
        self.widths, self.heights = map(set, zip(*sizes))

    @cached_property
    def max_width(self) -> int:
        """
        Compute the maximum width of images
        """
        return max(self.widths)

    @cached_property
    def is_fixed_height(self) -> bool:
        """
        Check if all images have the same height
        """
        return len(self.heights) == 1
