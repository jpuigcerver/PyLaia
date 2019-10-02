import warnings

from laia.data.transforms.vision import ToImageTensor


# TODO: Remove this
class ImageToTensor(ToImageTensor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of laia.utils.ImageToTensor is deprecated, "
            "please use laia.data.transforms.vision.ToImageTensor instead."
        )
        super().__init__(*args, **kwargs)
