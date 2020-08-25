from typing import Optional, Callable

import torch
import torchvision
from PIL import ImageOps, Image


class Invert:
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img: Image) -> Image:
        return ImageOps.invert(img)

    def __repr__(self) -> str:
        return "vision.{}()".format(self.__class__.__name__)


class Convert:
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode: str) -> None:
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img: Image) -> Image:
        return img.convert(self.mode)

    def __repr__(self) -> str:
        format_string = "vision." + self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += "mode={}".format(self.mode)
        format_string += ")"
        return format_string


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
        self._convert_transform = Convert(mode)
        self._should_invert = invert
        self._invert_transform = Invert() if self._should_invert else lambda x: x
        self._random_transform = (
            lambda x: random_transform(x) if random_transform else x
        )
        self._should_resize = fixed_height or fixed_width
        self._resize_transform = (
            lambda x: self.resize_transform(x, fixed_height, fixed_width)
            if self._should_resize
            else x
        )
        self._should_pad = min_height or min_width
        self._pad_transform = (
            lambda x: self.pad_transform(x, min_height, min_width, pad_color=pad_color)
            if self._should_pad
            else x
        )
        self._tensor_transform = ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        assert isinstance(img, Image.Image)
        img = self._convert_transform(img)
        img = self._invert_transform(img)
        img = self._random_transform(img)
        img = self._resize_transform(img)
        img = self._pad_transform(img)
        img = self._tensor_transform(img)
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
        return self.__class__.__name__ + (
            f"(\n  {self._convert_transform},\n  "
            + (f"{self._invert_transform},\n  " if self._should_invert else "")
            + ("vision.resize_transform(),\n  " if self._should_resize else "")
            + ("vision.pad_transform(),\n  " if self._should_pad else "")
            + f"{self._tensor_transform}\n)"
        )


ToTensor = torchvision.transforms.transforms.ToTensor
