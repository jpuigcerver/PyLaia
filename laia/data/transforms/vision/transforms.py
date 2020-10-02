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
        return f"vision.{self.__class__.__name__}(mode={self.mode})"


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
            self.resize_transform if fixed_width or fixed_height else None
        )
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        self.pad_transform = self.pad_transform if min_width or min_height else None
        self.min_width = min_width
        self.min_height = min_height
        self.pad_color = pad_color
        self.tensor_transform = ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # W x H
        assert isinstance(img, Image.Image)
        img = self.convert_transform(img)
        if self.invert_transform:
            img = self.invert_transform(img)
        if self.random_transform:
            img = self.random_transform(img)
        if self.resize_transform:
            img = self.resize_transform(img, fw=self.fixed_width, fh=self.fixed_height)
        if self.pad_transform:
            img = self.pad_transform(
                img, mw=self.min_width, mh=self.min_height, pad_color=self.pad_color
            )
        img = self.tensor_transform(img)
        # C x H x W
        return img

    @staticmethod
    def resize_transform(
        img: Image.Image,
        fw: Optional[int] = None,
        fh: Optional[int] = None,
        resample: int = Image.ANTIALIAS,
    ) -> Image.Image:
        if fw and fh:
            # resize to a fixed size
            return img.resize((fw, fh), resample=resample)
        w, h = img.size
        if fw is None:
            fw = w
        if fh is None:
            fh = h
        if fw > w:
            # upscale to a fixed width
            return img.resize((fw, h * fw // w), resample=resample)
        if fh > h:
            # upscale to a fixed height
            return img.resize((w * fh // h, fh), resample=resample)
        # downscale
        img.thumbnail((fw, fh), resample=resample)
        return img

    @staticmethod
    def pad_transform(
        img: Image.Image,
        mw: Optional[int] = None,
        mh: Optional[int] = None,
        pad_color: int = 0,
    ) -> Image.Image:
        w, h = img.size
        nw = w if mw is None or w > mw else mw
        nh = h if mh is None or h > mh else mh
        # maybe an extra pixel to the bottom
        a, b = divmod(nw - w, 2)
        left, right = a, a + b
        # maybe an extra pixel to the right
        a, b = divmod(nh - h, 2)
        top, bottom = a, a + b
        return torchvision.transforms.functional.pad(
            img, (left, top, right, bottom), fill=pad_color
        )

    def __repr__(self) -> str:
        invert_txt = f"{self.invert_transform},\n  " if self.invert_transform else ""
        random_txt = f"{self.random_transform},\n  " if self.random_transform else ""
        return (
            f"{self.__class__.__name__}(\n  "
            f"{self.convert_transform},\n  "
            f"{invert_txt}"
            f"{random_txt}"
            + ("vision.resize_transform(),\n  " if self.resize_transform else "")
            + ("vision.pad_transform(),\n  " if self.pad_transform else "")
            + f"{self.tensor_transform}\n)"
        )


ToTensor = torchvision.transforms.transforms.ToTensor
