import torchvision

from laia.data import PaddedTensor


class Feeder:
    """This class is used to feed data to a model or loss."""

    def __call__(self, x):
        return self.feed(x)

    def feed(self, x):
        raise NotImplementedError("Abstract class.")


class ItemFeeder(Feeder):
    """Feed an element from a dictionary, by its key.

    Args:
      key: the key to use.
    """

    def __init__(self, key):
        super().__init__()
        self._key = key

    def feed(self, x):
        assert x is not None
        assert self._key in x, f"Could not find key {self._key} in {x}"
        return x[self._key]


class ImageFeeder(Feeder):
    """Feed an image as a PyTorch tensor.

    Args:
      keep_padded_tensors: Whether or not keep the size
          information of the padding. If False, the batch tensor will be
          returned without any size information. (default: True)
      keep_channels_in_size: Whether or not the number of channels of the
          images is kept as part of the size in the `PaddedTensor` objects.
    """

    def __init__(
        self, keep_padded_tensors: bool = True, keep_channels_in_size: bool = False
    ) -> None:
        super().__init__()
        self._keep_padded_tensors = keep_padded_tensors
        self._keep_channels_in_size = keep_channels_in_size

    @classmethod
    def view_as_4d(cls, x):
        if x.dim() > 4:
            raise ValueError(
                f"Tensor with {x.dim()} dimensions is not supported as an Image"
            )
        while x.dim() <= 3:
            x = x.unsqueeze(0)
        return x

    def feed(self, x):
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
        elif isinstance(x, tuple) and len(x) == 2:
            # required because of: https://github.com/pytorch/pytorch/issues/44009
            # can be deleted when we no longer support torch<=1.6.0
            x, xs = x
        else:
            x, xs = x, None
        x = self.view_as_4d(x)  # N x C x H x W
        if xs is not None and self._keep_padded_tensors:
            if xs.size(1) == 3 and not self._keep_channels_in_size:
                xs = xs[:, 1:]
            return PaddedTensor.build(x, xs)
        else:
            return x


Compose = torchvision.transforms.Compose
