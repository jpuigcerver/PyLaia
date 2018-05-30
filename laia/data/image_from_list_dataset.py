from __future__ import absolute_import

from io import open

from torch._six import string_classes

import laia.logging as log
from laia.data.image_dataset import ImageDataset
from laia.data.text_image_from_text_table_dataset import (
    find_image_filename_from_id,
    IMAGE_EXTENSIONS,
)

_logger = log.get_logger(__name__)


class ImageFromListDataset(ImageDataset):

    def __init__(
        self,
        img_list,
        img_dirs=None,
        img_transform=None,
        img_extensions=IMAGE_EXTENSIONS,
    ):
        self._ids, imgs = _get_ids_and_images_from_img_list(
            img_list, img_dirs, img_extensions
        )
        super(ImageFromListDataset, self).__init__(imgs, img_transform)

    def __getitem__(self, index):
        """Returns the ID of the example, and its image.

        Args:
          index (int): Index of the item to return.

        Returns:
          dict: Dictionary containing the example ID ('id'), image ('img'),
        """
        out = super(ImageFromListDataset, self).__getitem__(index)
        out["id"] = self._ids[index]
        return out


def _load_image_list_from_file(img_list, encoding="utf-8"):
    if isinstance(img_list, string_classes):
        f = open(img_list, "r", encoding=encoding)
        img_list = [i.rstrip() for i in f.readlines()]
        f.close()
    return img_list


def _get_ids_and_images_from_img_list(img_list, img_dirs, img_extensions):
    img_list = _load_image_list_from_file(img_list)
    ids, imgs = [], []
    for imgid in img_list:
        imgid = imgid.rstrip()
        if img_dirs is None:
            # If no img directory is specified, then img_list
            # must contain whole paths to the images
            fname = imgid
        else:
            for dir in img_dirs:
                fname = find_image_filename_from_id(imgid, dir, img_extensions)
                if fname is not None:
                    break
        if fname is None:
            _logger.warning(
                "No image file was found " 'for image ID "{}", ignoring example...',
                imgid,
            )
            continue
        else:
            ids.append(imgid)
            imgs.append(fname)
    return ids, imgs
