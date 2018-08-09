from __future__ import absolute_import

import numpy as np
import torch
from PIL import Image

from laia.data.transformers.transformer import Transformer


class TransformerImageTensor(Transformer):
    """Convert a PIL image to a PyTorch tensor with C x H x W layout."""

    def __init__(self):
        super(TransformerImageTensor, self).__init__()

    def __call__(self, x):
        # type: (Image) -> Image
        x = np.asarray(x, dtype=np.float32)
        if len(x.shape) != 3:
            x = np.expand_dims(x, axis=-1)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x / 255.0)
