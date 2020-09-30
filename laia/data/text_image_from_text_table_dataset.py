from os import listdir
from os.path import isfile, join, splitext
from typing import Callable, Generator, List, Sequence, TextIO, Tuple, Union

import laia.common.logging as log
from laia.data.text_image_dataset import TextImageDataset

IMAGE_EXTENSIONS = ".jpg", ".png", ".jpeg", ".pbm", ".pgm", ".ppm", ".bmp"

_logger = log.get_logger(__name__)


class TextImageFromTextTableDataset(TextImageDataset):
    def __init__(
        self,
        txt_table: str,
        img_dirs: Union[List[str], str],
        img_transform: Callable = None,
        txt_transform: Callable = None,
        img_extensions: Sequence[str] = IMAGE_EXTENSIONS,
        encoding: str = "utf-8",
    ):
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]
        # First, load the transcripts and find the corresponding image filenames
        # in the given directory. Also save the IDs (basename) of the examples.
        self._ids, imgs, txts = _get_images_and_texts_from_text_table(
            txt_table, img_dirs, img_extensions, encoding=encoding
        )
        # Prepare dataset using the previous image filenames and transcripts.
        super().__init__(imgs, txts, img_transform, txt_transform)

    def __getitem__(self, index: int):
        """Returns the ID of the example, the image and its transcript from
        the dataset.

        Args:
          index (int): Index of the item to return.

        Returns:
          dict: Dictionary containing the example ID ('id'), image ('img') and
            the transcript ('txt') of the image.
        """
        out = super().__getitem__(index)
        out["id"] = self._ids[index]
        return out


def _get_valid_image_filenames_from_dir(img_dir: str, img_extensions: Sequence[str]):
    img_extensions = set(img_extensions)
    valid_image_filenames = {}
    for fname in listdir(img_dir):
        bname, ext = splitext(fname)
        fname = join(img_dir, fname)
        if isfile(fname) and ext.lower() in img_extensions:
            valid_image_filenames[bname] = fname
    return valid_image_filenames


def find_image_filepath_from_id(
    imgid: str, img_dir: str, img_extensions: Sequence[str]
):
    extensions = set(ext.lower() for ext in img_extensions)
    extensions.update(ext.upper() for ext in img_extensions)
    for ext in extensions:
        fname = join(img_dir, imgid if imgid.endswith(ext) else imgid + ext)
        if isfile(fname):
            return fname
    return None


def _load_text_table_from_file(
    table_file: Union[TextIO, str], encoding: str = "utf-8"
) -> Generator[Tuple[int, str, str], None, None]:
    if isinstance(table_file, str):
        table_file = open(table_file, encoding=encoding)
    for n, line in enumerate((l.split(maxsplit=1) for l in table_file), 1):
        # Skip empty lines and lines starting with #
        if not len(line) or line[0].startswith("#"):
            continue
        yield n, line[0], line[1]
    table_file.close()


def _get_images_and_texts_from_text_table(
    table_file: Union[TextIO, str],
    img_dirs: Sequence[str],
    img_extensions: Sequence[str],
    encoding="utf-8",
) -> Tuple[List[str], List[str], List[str]]:
    assert len(img_dirs) > 0, "No image directory provided"
    ids, imgs, txts = [], [], []
    for _, imgid, txt in _load_text_table_from_file(table_file, encoding=encoding):
        imgid = imgid.rstrip()
        for dir in img_dirs:
            fname = find_image_filepath_from_id(imgid, dir, img_extensions)
            if fname is not None:
                break
        if fname is None:
            _logger.warning(
                "No image file was found for image ID '{}', ignoring example...", imgid
            )
            continue
        else:
            ids.append(imgid)
            imgs.append(fname)
            txts.append(txt)

    return ids, imgs, txts
