from typing import Any, Dict, List

import torch

from laia.losses.ctc_loss import transform_batch


class CTCGreedyDecoder:
    def __call__(self, x: Any, segmentation: bool = False) -> Dict[str, List]:
        x, xs = transform_batch(x)
        x = x.detach()
        x = [x[: xs[i], i, :] for i in range(len(xs))]
        x = [x_n.max(dim=1) for x_n in x]
        out = {}
        if segmentation:
            out["prob"] = [x_n.values.exp() for x_n in x]
            out["segmentation"] = [
                CTCGreedyDecoder.compute_segmentation(x_n.indices.tolist()) for x_n in x
            ]
        x = [x_n.indices for x_n in x]
        # Remove repeated symbols
        x = [torch.unique_consecutive(x_n) for x_n in x]
        # Remove CTC blank symbol
        x = [x_n[torch.nonzero(x_n, as_tuple=True)] for x_n in x]
        out["hyp"] = [x_n.tolist() for x_n in x]
        return out

    @staticmethod
    def compute_segmentation(x: List[int]) -> List[int]:
        if not x:
            return []
        return [0] + [i for i in range(1, len(x)) if x[i] != x[i - 1] != 0] + [len(x)]
