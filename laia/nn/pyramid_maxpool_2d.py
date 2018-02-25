from __future__ import absolute_import

import torch
from nnutils_pytorch import adaptive_maxpool_2d

from laia.data import PaddedTensor


class PyramidMaxPool2d(torch.nn.Module):
    def __init__(self, levels, vertical=False):
        super(PyramidMaxPool2d, self).__init__()
        assert levels > 0
        self._levels = levels
        self._vertical = vertical

    def forward(self, x):
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
        else:
            xs = None

        n, c, _, _ = x.size()

        out_levels = []
        for level in range(1, self._levels + 1):
            if self._vertical:
                output_sizes = (level, 1)
            else:
                output_sizes = (1, level)

            y = adaptive_maxpool_2d(batch_input=x,
                                    output_sizes=output_sizes,
                                    batch_sizes=xs)
            out_levels.append(y.view(n, c * level))

        return torch.cat(out_levels, dim=1)
