from __future__ import absolute_import

from os import listdir
from os.path import isfile, join, splitext

import laia.logging as log
from laia.data.text_image_dataset import TextImageDataset
from torch._six import string_classes

_IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.pbm', '.pgm', '.ppm', '.bmp')

_logger = log.get_logger(__name__)


class TextImageFromTextTableDataset(TextImageDataset):
    def __init__(self, txt_table, imgs_dir, img_transform=None,
                 txt_transform=None, img_extensions=_IMAGE_EXTENSIONS):
        # First, load the transcripts and find the corresponding image filenames
        # in the given directory. Also save the IDs (basename) of the examples.
        self._ids, imgs, txts = _get_images_and_texts_from_text_table(
            txt_table, imgs_dir, img_extensions)
        # Prepare dataset using the previous image filenames and transcripts.
        super(TextImageFromTextTableDataset, self).__init__(
            imgs, txts, img_transform, txt_transform)

    def __getitem__(self, index):
        """Returns the ID of the example, the image and its transcript from
        the dataset.
        Args:
          index (int): Index of the item to return.

        Returns:
          dict: Dictionary containing the example ID ('id'), image ('img') and
            the transcript ('txt') of the image.
        """
        out = super(TextImageFromTextTableDataset, self).__getitem__(index)
        out['id'] = self._ids[index]
        return out


def _get_valid_image_filenames_from_dir(imgs_dir, img_extensions):
    img_extensions = set(img_extensions)
    valid_image_filenames = {}
    for fname in listdir(imgs_dir):
        bname, ext = splitext(fname)
        fname = join(imgs_dir, fname)
        if isfile(fname) and ext.lower() in img_extensions:
            valid_image_filenames[bname] = fname
    return valid_image_filenames


def _find_image_filename_from_id(img_id, img_dir, img_extensions):
    img_extensions = set(img_extensions)
    for ext in img_extensions:
        for ext in [ext.lower(), ext.upper()]:
            fname = join(img_dir, img_id + ext)
            if isfile(fname):
                return fname
    return None


def _load_text_table_from_file(table_file):
    if isinstance(table_file, string_classes):
        table_file = open(table_file, 'r')

    for n, line in enumerate(table_file, 1):
        line = line.split()
        # Skip empty lines and lines starting with #
        if len(line) == 0 or line[0][0] == '#':
            continue
        yield n, line[0], line[1:]

    table_file.close()


def _get_images_and_texts_from_text_table(table_file, imgs_dir, img_extensions):
    ids, imgs, txts = [], [], []
    for _, imgid, txt in _load_text_table_from_file(table_file):
        fname = _find_image_filename_from_id(imgid, imgs_dir, img_extensions)
        if fname is None:
            _logger.warning('No image file was found in folder "{}" for image '
                            'ID "{}", ignoring example...', imgs_dir, imgid)
            continue
        else:
            ids.append(imgid)
            imgs.append(fname)
            txts.append(txt)

    return ids, imgs, txts
