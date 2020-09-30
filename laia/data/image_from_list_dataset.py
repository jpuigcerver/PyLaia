from os.path import isfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import laia.common.logging as log
from laia.data import ImageDataset
from laia.data.text_image_from_text_table_dataset import (
    IMAGE_EXTENSIONS,
    find_image_filepath_from_id,
)

_logger = log.get_logger(__name__)


class ImageFromListDataset(ImageDataset):
    def __init__(
        self,
        img_list: Union[str, List[str]],
        img_dirs: Optional[List[str]] = None,
        img_transform: Callable = None,
        img_extensions: List[str] = IMAGE_EXTENSIONS,
    ):
        self._ids, imgs = _get_img_ids_and_filepaths(
            img_list, img_dirs=img_dirs, img_extensions=img_extensions
        )
        super().__init__(imgs, img_transform)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns the ID of the example, and its image.

        Args:
          index (int): Index of the item to return.

        Returns:
          dict: Dictionary containing the example ID ('id'), image ('img'),
        """
        out = super().__getitem__(index)
        out["id"] = self._ids[index]
        return out


def _load_image_list_from_file(img_list: Union[str, List[str]]) -> List[str]:
    if isinstance(img_list, str):
        with open(img_list) as f:
            img_list = [l.strip() for l in f]
            # skip empty lines and lines starting with '#'
            img_list = [l for l in img_list if l and not l.startswith("#")]
    return img_list


def _get_img_ids_and_filepaths(
    img_list: Union[str, List[str]],
    img_dirs: Optional[List[str]] = None,
    img_extensions: List[str] = IMAGE_EXTENSIONS,
) -> Tuple[List[str], List[str]]:
    if img_dirs is None:
        img_dirs = []
    img_list = _load_image_list_from_file(img_list)
    ids, filepaths = [], []
    for img_id in img_list:
        img_id = img_id.strip()
        for dir in img_dirs:
            filepath = find_image_filepath_from_id(img_id, dir, img_extensions)
            if filepath is not None:
                break
        else:
            if isfile(img_id):
                # img_list must contain whole paths to the images
                filepath = img_id
            else:
                _logger.warning(
                    'No image file was found for image ID "{}", ignoring example...',
                    img_id,
                )
                continue
        ids.append(img_id)
        filepaths.append(filepath)
    return ids, filepaths
