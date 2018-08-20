import warnings

from laia.data.transforms.text import ToTensor


# TODO: Remove this
class TextToTensor(ToTensor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of laia.utils.TextToTensor is deprecated, "
            "please use laia.data.transforms.text.ToTensor instead."
        )
        super(TextToTensor, self).__init__(*args, **kwargs)
