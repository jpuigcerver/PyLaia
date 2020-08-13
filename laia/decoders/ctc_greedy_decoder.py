from typing import List

import torch

from laia.losses.ctc_loss import transform_output


class CTCGreedyDecoder:
    def __init__(self):
        self._output = None
        self._segmentation = None

    def __call__(self, x, segmentation=False):
        x, xs = transform_output(x)
        idx = x.argmax(dim=2).t()
        x = [idx[i, :v] for i, v in enumerate(xs)]
        if segmentation:
            self._segmentation = [
                CTCGreedyDecoder.compute_segmentation(x_n) for x_n in x
            ]
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
    def segmentation(self):
        return self._segmentation

    @staticmethod
    def compute_segmentation(x: List[int]) -> List[int]:
        if not x:
            return []
        return [0] + [i for i in range(1, len(x)) if x[i] != x[i - 1] != 0] + [len(x)]
