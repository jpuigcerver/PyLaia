#!/usr/bin/env python3
from functools import cached_property
from pathlib import Path
from typing import List, Optional, TextIO, Union

import imagesize

from laia.data.image_from_list_dataset import _get_img_ids_and_filepaths
from laia.data.text_image_from_text_table_dataset import (
    _get_images_and_texts_from_text_table,
)


class ImageStats:
    """
    Compute statistics on the dataset

    Args:
        stage: String indicating the stage of the processing, either "test" or "fit"
        tr_txt_table: Path to the train text table (train mode)
        va_txt_table: Path to the validation text table (train mode)
        img_list: Path to the list of test images (test mode)
        img_dir: Path to images
    """

    def __init__(
        self,
        stage: str,
        tr_txt_table: Optional[Union[TextIO, str, List[str]]] = None,
        va_txt_table: Optional[Union[TextIO, str, List[str]]] = None,
        img_list: Optional[Union[TextIO, str, List[str]]] = None,
        img_dirs: Optional[Union[List[str], str, List[Path], Path]] = None,
    ):
        assert stage in ["fit", "test"]

        if stage == "fit":
            assert tr_txt_table and va_txt_table
            filenames = _get_images_and_texts_from_text_table(tr_txt_table, img_dirs)[1]
            filenames += _get_images_and_texts_from_text_table(va_txt_table, img_dirs)[
                1
            ]

        elif stage == "test":
            assert img_list
            filenames = _get_img_ids_and_filepaths(img_list, img_dirs)[1]

        sizes = list(map(imagesize.get, filenames))
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
