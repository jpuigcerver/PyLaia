from typing import Any, Dict, List

import torch

from laia.losses.ctc_loss import transform_batch


class CTCGreedyDecoder:
    def __call__(
        self, x: Any, segmentation: bool = False, apply_softmax: bool = True
    ) -> Dict[str, List]:
        x, xs = transform_batch(x)
        x = x.detach()

        # Apply softmax to have log-probabilities
        if apply_softmax:
            x = torch.nn.functional.log_softmax(x, dim=-1)

        # Get device
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        # Transform to list of size = batch_size
        x = [x[: xs[i], i, :] for i in range(len(xs))]
        x = [x_n.max(dim=1) for x_n in x]

        # If needed, return text <-> image alignments
        out = {}
        if segmentation:
            out["prob-segmentation"] = [x_n.values.exp() for x_n in x]
            out["segmentation"] = [
                CTCGreedyDecoder.compute_segmentation(x_n.indices.tolist()) for x_n in x
            ]

        # Get symbols and probabilities
        probs = [x_n.values.exp() for x_n in x]
        x = [x_n.indices for x_n in x]

        # Remove consecutive symbols
        # Keep track of counts of consecutive symbols. Example: [0, 0, 0, 1, 2, 2] => [3, 1, 2]
        counts = [torch.unique_consecutive(x_n, return_counts=True)[1] for x_n in x]

        # Select indexes to keep. Example: [0, 3, 4] (always keep the first index, then use cumulative sum of counts tensor)
        zero_tensor = torch.tensor([0], device=device)
        idxs = [torch.cat((zero_tensor, count.cumsum(0)[:-1])) for count in counts]

        # Keep only non consecutive symbols and their associated probabilities
        x = [x[i][idxs[i]] for i in range(len(x))]
        probs = [probs[i][idxs[i]] for i in range(len(x))]

        # Remove blank symbols
        # Get index for non blank symbols
        idxs = [torch.nonzero(x_n, as_tuple=True) for x_n in x]

        # Keep only non blank symbols and their associated probabilities
        x = [x[i][idxs[i]] for i in range(len(x))]
        probs = [probs[i][idxs[i]] for i in range(len(x))]

        # Save results
        out["hyp"] = [x_n.tolist() for x_n in x]

        # Return char-based probability
        out["prob-htr-char"] = [prob.tolist() for prob in probs]
        return out

    @staticmethod
    def compute_segmentation(x: List[int]) -> List[int]:
        if not x:
            return []
        return [0] + [i for i in range(1, len(x)) if x[i] != x[i - 1] != 0] + [len(x)]
