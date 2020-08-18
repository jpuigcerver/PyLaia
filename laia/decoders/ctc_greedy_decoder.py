from typing import List

import torch

from laia.losses.ctc_loss import transform_output


class CTCGreedyDecoder:
    def __init__(self):
        self._output = None
        self._prob = None
        self._segmentation = None

    def __call__(self, x, segmentation=False):
        x, xs = transform_output(x)
        x = x.detach()
        x = [x[: xs[i], i, :] for i in range(len(xs))]
        x = [x_n.max(dim=1) for x_n in x]
        if segmentation:
            self._prob = [x_n.values.exp() for x_n in x]
            self._segmentation = [
                CTCGreedyDecoder.compute_segmentation(x_n.indices.tolist()) for x_n in x
            ]
        x = [x_n.indices for x_n in x]
        # Remove repeated symbols
        x = [torch.unique_consecutive(x_n) for x_n in x]
        # Remove CTC blank symbol
        x = [x_n[torch.nonzero(x_n, as_tuple=True)] for x_n in x]
        self._output = [x_n.tolist() for x_n in x]
        return self._output

    @property
    def output(self):
        return self._output

    @property
    def prob(self):
        return self._prob

    @property
    def segmentation(self):
        return self._segmentation

    @staticmethod
    def compute_segmentation(x: List[int]) -> List[int]:
        if not x:
            return []
        return [0] + [i for i in range(1, len(x)) if x[i] != x[i - 1] != 0] + [len(x)]
