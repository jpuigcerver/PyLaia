import torchvision
from PIL import ImageOps, Image


class Invert(object):
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img):
        # type: (Image) -> Image
        return ImageOps.invert(img)

    def __repr__(self):
        return "vision.{}()".format(self.__class__.__name__)


class Convert(object):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode):
        # type: (str) -> None
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img):
        # type: (Image) -> Image
        return img.convert(self.mode)

    def __repr__(self):
        format_string = "vision." + self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += "mode={}".format(self.mode)
        format_string += ")"
        return format_string


ToTensor = torchvision.transforms.transforms.ToTensor
