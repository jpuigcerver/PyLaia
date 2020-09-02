from typing import Callable, Optional

import torch
import torchvision
from PIL import Image, ImageOps


class Invert:
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img: Image) -> Image:
        return ImageOps.invert(img)

    def __repr__(self) -> str:
        return f"vision.{self.__class__.__name__}()"


class Convert:
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode: str) -> None:
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img: Image) -> Image:
        return img.convert(self.mode)

    def __repr__(self) -> str:
        return f"vision.{self.__class__.__name__}({f'mode={self.mode}' if self.mode is not None else ''})"


class ToImageTensor:
    def __init__(
        self,
        invert: bool = True,
        mode: str = "L",
        random_transform: Optional[Callable] = None,
        fixed_height: Optional[int] = None,
        fixed_width: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        pad_color: int = 0,
    ) -> None:
        assert mode in ("L", "RGB", "RGBA")
        assert fixed_height is None or fixed_height > 0
        assert fixed_width is None or fixed_width > 0
        assert min_height is None or min_height > 0
        assert min_width is None or min_width > 0
        self.convert_transform = Convert(mode)
        self.invert_transform = Invert() if invert else None
        self.random_transform = random_transform if random_transform else None
        self.resize_transform = (
            self.resize_transform if fixed_height or fixed_width else None
        )
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        self.pad_transform = self.pad_transform if min_height or min_width else None
        self.min_height = min_width
        self.min_width = min_width
        self.pad_color = pad_color
        self.tensor_transform = ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        assert isinstance(img, Image.Image)
        img = self.convert_transform(img)
        if self.invert_transform:
            img = self.invert_transform(img)
        if self.random_transform:
            img = self.random_transform(img)
        if self.resize_transform:
            img = self.resize_transform(img, self.fixed_height, self.fixed_width)
        if self.pad_transform:
            img = self.pad_transform(
                img, self.min_height, self.min_width, pad_color=self.pad_color
            )
        img = self.tensor_transform(img)
        # C x H x W
        return img

    @staticmethod
    def resize_transform(
        img: Image.Image, fh: Optional[int], fw: Optional[int]
    ) -> Image.Image:
        w, h = img.size
        nh = fh if fh else h * fw // w
        nw = fw if fw else w * fh // h
        return img.resize((nw, nh), Image.BILINEAR)

    @staticmethod
    def pad_transform(
        img: Image.Image, mh: Optional[int], mw: Optional[int], pad_color: int = 0
    ) -> Image.Image:
        w, h = img.size
        nh = h if mh is None or h >= mh else mh
        nw = w if mw is None or w >= mw else mw
        return torchvision.transforms.functional.pad(
            img, ((nw - w) // 2, (nh - h) // 2), fill=pad_color
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n  "
            f"{self.convert_transform},\n  "
            f"{self.invert_transform if self.invert_transform else ''},\n  "
            f"{self.random_transform if self.random_transform else ''},\n  "
            + ("vision.resize_transform(),\n  " if self.resize_transform else "")
            + ("vision.pad_transform(),\n  " if self.pad_transform else "")
            + f"{self.tensor_transform}\n)"
        )


ToTensor = torchvision.transforms.transforms.ToTensor
