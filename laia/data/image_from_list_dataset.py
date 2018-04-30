from __future__ import absolute_import

from torch._six import string_classes

from laia.data.image_dataset import ImageDataset
from laia.data.text_image_from_text_table_dataset import find_image_filename_from_id, IMAGE_EXTENSIONS


class ImageFromListDataset(ImageDataset):
    def __init__(self, img_list, img_dir=None, img_transform=None, img_extensions=IMAGE_EXTENSIONS):
        if isinstance(img_list, string_classes):
            with open(img_list, 'r') as f:
                img_list = [i.rstrip() for i in f.readlines()]
        # If img_dir is None then img_list must contain whole paths to the images
        imgs = img_list if img_dir is None else \
            [find_image_filename_from_id(id_, img_dir, img_extensions)
             for id_ in img_list]
        super(ImageFromListDataset, self).__init__(imgs, img_transform)
