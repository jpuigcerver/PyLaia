from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, imgs: List[str], transform: Optional[Callable[[Image.Image], Any]] = None
    ):
        assert isinstance(imgs, (list, tuple))
        super().__init__()
        self._imgs = imgs
        self._transform = transform

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns a dictionary containing the given image from the dataset.
        The image is associated with the key 'img'."""
        img = Image.open(self._imgs[index])
        if self._transform:
            img = self._transform(img)
        return {"img": img}

    def __len__(self) -> int:
        return len(self._imgs)
