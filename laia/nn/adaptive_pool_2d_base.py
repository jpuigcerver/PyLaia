from __future__ import absolute_import

import torch

from laia.data import PaddedTensor


class AdaptivePool2dBase(torch.nn.Module):
    def __init__(self, output_sizes, func):
        super(AdaptivePool2dBase, self).__init__()
        self._output_sizes = output_sizes
        self._func = func
        self._fixed_size = isinstance(output_sizes, int) or (
            output_sizes[0] is not None and output_sizes[1] is not None
        )

    @property
    def output_sizes(self):
        return self._output_sizes

    def forward(self, x):
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        y = self._func(batch_input=x, output_sizes=self.output_sizes, batch_sizes=xs)
        if xs is None or self._fixed_size:
            return y
        else:
            ys = xs.clone()
            if self.output_sizes[0] is not None:
                ys[:, 0] = self.output_sizes[0]
            else:
                ys[:, 1] = self.output_sizes[1]
            return PaddedTensor(data=y, sizes=ys)
