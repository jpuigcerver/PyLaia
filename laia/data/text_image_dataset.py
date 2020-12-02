from typing import Any, Callable, Dict, List

from laia.data import ImageDataset


class TextImageDataset(ImageDataset):
    def __init__(
        self,
        imgs: List[str],
        txts: List[str],
        img_transform: Callable = None,
        txt_transform: Callable = None,
    ):
        super().__init__(imgs, img_transform)
        assert len(imgs) == len(txts)
        self._txts = txts
        self._txt_transform = txt_transform

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns an image and its transcript from the dataset."""
        # Get image
        out = super().__getitem__(index)
        # Get transcript
        txt = self._txts[index]
        if self._txt_transform:
            txt = self._txt_transform(txt)
        # Return image and transcript
        out["txt"] = txt
        return out
