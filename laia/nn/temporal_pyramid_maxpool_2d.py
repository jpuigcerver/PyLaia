import warnings
from typing import Union, Sequence

import torch

from laia.data import PaddedTensor

try:
    from nnutils_pytorch import adaptive_maxpool_2d
except ImportError:
    adaptive_maxpool_2d = None


def _adaptive_maxpool_2d(batch_input, output_sizes, batch_sizes, use_nnutils):
    if adaptive_maxpool_2d and use_nnutils:
        return adaptive_maxpool_2d(
            batch_input=batch_input, output_sizes=output_sizes, batch_sizes=batch_sizes
        )
    else:
        if use_nnutils:
            warnings.warn(
                "You are trying to use nnutils_pytorch.adaptive_maxpool_2d "
                "but nnutils_pytorch is not installed. Install the package "
                "to avoid this warning."
            )

        if batch_sizes is None:
            return torch.nn.functional.adaptive_max_pool2d(
                input=batch_input, output_size=output_sizes
            )
        else:
            to_stack = []
            for n in range(batch_input.size(0)):
                nh, nw = int(batch_sizes[n, 0]), int(batch_sizes[n, 1])
                batch_view = batch_input[n, :, :nh, :nw].contiguous()
                to_stack.append(
                    torch.nn.functional.adaptive_max_pool2d(
                        input=batch_view, output_size=output_sizes
                    )
                )
            return torch.stack(to_stack)


class TemporalPyramidMaxPool2d(torch.nn.Module):
    def __init__(self, levels, vertical=False, use_nnutils=True):
        # type: (Sequence[int], bool, bool) -> None
        super(TemporalPyramidMaxPool2d, self).__init__()
        self._levels = levels
        self._vertical = vertical
        self._use_nnutils = use_nnutils

    def forward(self, x):
        # type: (Union[torch.Tensor, PaddedTensor]) -> torch.Tensor
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
        else:
            xs = None

        n, c, _, _ = x.size()

        out_levels = []
        for level in self._levels:
            output_sizes = (level, 1) if self._vertical else (1, level)
            y = _adaptive_maxpool_2d(
                batch_input=x,
                output_sizes=output_sizes,
                batch_sizes=xs,
                use_nnutils=self._use_nnutils,
            )
            out_levels.append(y.view(n, c * level))

        return torch.cat(out_levels, dim=1)
