from os.path import isfile, join
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TextIO, Tuple, Union

import laia.common.logging as log
from laia.data.text_image_dataset import TextImageDataset

IMAGE_EXTENSIONS = ".jpg", ".png", ".jpeg", ".pbm", ".pgm", ".ppm", ".bmp"

_logger = log.get_logger(__name__)


class TextImageFromTextTableDataset(TextImageDataset):
    def __init__(
        self,
        txt_table: Union[TextIO, str, List[str]],
        img_dirs: Optional[Union[List[str], str, List[Path], Path]] = None,
        img_transform: Callable = None,
        txt_transform: Callable = None,
        img_extensions: List[str] = IMAGE_EXTENSIONS,
    ):
        if img_dirs is None:
            img_dirs = []
        elif isinstance(img_dirs, (str, Path)):
            img_dirs = [img_dirs]
        # First, load the transcripts and find the corresponding image filenames
        # in the given directory. Also save the IDs (basename) of the examples.
        self._ids, imgs, txts = _get_images_and_texts_from_text_table(
            txt_table, img_dirs=img_dirs, img_extensions=img_extensions
        )
        # Prepare dataset using the previous image filenames and transcripts.
        super().__init__(imgs, txts, img_transform, txt_transform)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns the ID of the example, the image and its transcript from
        the dataset.

        Args:
          index: Index of the item to return.

        Returns:
          Dictionary containing the example ID ('id'), image ('img') and
            the transcript ('txt') of the image.
        """
        out = super().__getitem__(index)
        out["id"] = self._ids[index]
        return out


def find_image_filepath_from_id(
    img_id: str, img_dir: Union[str, Path], img_extensions: List[str] = IMAGE_EXTENSIONS
) -> Optional[str]:
    extensions = set(ext.lower() for ext in img_extensions)
    extensions.update(ext.upper() for ext in img_extensions)
    for ext in extensions:
        filepath = join(img_dir, img_id if img_id.endswith(ext) else img_id + ext)
        if isfile(filepath):
            return filepath
    return


def _load_text_table_from_file(
    table_file: Union[TextIO, str, List[str], Path],
) -> Generator[Tuple[int, str, str], None, None]:
    if isinstance(table_file, (str, Path)):
        table_file = open(table_file)
    for line in (l.split(maxsplit=1) for l in table_file):
        # skip empty lines and lines starting with '#'
        if not line or line[0].startswith("#"):
            continue
        elif len(line) == 1:
            _logger.warning(
                "No text found for image ID '{}', ignoring example...", line[0]
            )
            continue
        img_id, txt = line
        img_id = img_id.strip()
        txt = txt.rstrip()
        yield img_id, txt
    if hasattr(table_file, "close"):
        table_file.close()


def _get_images_and_texts_from_text_table(
    table_file: Union[TextIO, str, List[str]],
    img_dirs: Optional[List[Union[str, Path]]] = None,
    img_extensions: List[str] = IMAGE_EXTENSIONS,
) -> Tuple[List[str], List[str], List[str]]:
    if img_dirs is None:
        img_dirs = []
    ids, filepaths, txts = [], [], []
    for img_id, txt in _load_text_table_from_file(table_file):
        for dir in img_dirs:
            filepath = find_image_filepath_from_id(
                img_id, dir, img_extensions=img_extensions
            )
            if filepath is not None:
                break
        else:
            if isfile(img_id):
                # the img id must be a path to the image
                filepath = img_id
            else:
                _logger.warning(
                    "No image file found for image ID '{}', ignoring example...", img_id
                )
                continue
        ids.append(img_id)
        filepaths.append(filepath)
        txts.append(txt)
    return ids, filepaths, txts
