#!/usr/bin/env python3
import math
from pathlib import Path
from typing import List, TextIO

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
        img_dirs: List[str | Path] = None,
        tr_txt_table: TextIO | str | List[str] = None,
        va_txt_table: TextIO | str | List[str] = None,
    ):
        self.tr_image_paths = _get_images_and_texts_from_text_table(
            tr_txt_table, img_dirs
        )[1]
        self.va_image_paths = _get_images_and_texts_from_text_table(
            va_txt_table, img_dirs
        )[1]
        self.compute_image_size_stats()

    def compute_image_size_stats(self) -> int:
        """
        Compute the maximum width of training and validation images
        """
        self.max_width = 0
        self.max_height = 0
        self.min_height = math.inf
        for img_path in self.tr_image_paths + self.va_image_paths:
            width, height = imagesize.get(img_path)
            self.max_width = max(self.max_width, width)
            self.max_height = max(self.max_height, height)
            self.min_height = min(self.min_height, height)
        self.is_fixed_height = self.min_height == self.max_height
