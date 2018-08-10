import torch
from nnutils_pytorch import mask_image_from_size

from laia.data import PaddedTensor


class MaskImageFromSize(torch.nn.Module):
    def __init__(self, mask_value=0, inplace=False):
        super(MaskImageFromSize, self).__init__()
        self.inplace = inplace
        self.mask_value = mask_value

    def forward(self, x):
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
            y = mask_image_from_size(
                batch_input=x,
                batch_sizes=xs,
                mask_value=self.mask_value,
                inplace=self.inplace,
            )
            return PaddedTensor(y, xs)
        else:
            return x
