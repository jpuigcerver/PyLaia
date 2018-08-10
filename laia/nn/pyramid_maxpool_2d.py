from typing import Sequence, Union

import torch

from laia.data import PaddedTensor
from laia.nn.temporal_pyramid_maxpool_2d import _adaptive_maxpool_2d


class PyramidMaxPool2d(torch.nn.Module):
    def __init__(self, levels: Sequence[int], use_nnutils: bool = True) -> None:
        super().__init__()
        self._levels = tuple(levels)
        self._use_nnutils = use_nnutils

    def forward(self, x: Union[torch.Tensor, PaddedTensor]) -> torch.Tensor:
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
        else:
            xs = None

        n, c, _, _ = x.size()

        out_levels = []
        for level in self._levels:
            size = 2 ** (level - 1)
            y = _adaptive_maxpool_2d(
                batch_input=x,
                output_sizes=(size, size),
                batch_sizes=xs,
                use_nnutils=self._use_nnutils,
            )
            out_levels.append(y.view(n, c * size * size))

        return torch.cat(out_levels, dim=1)
