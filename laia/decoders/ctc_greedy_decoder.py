from itertools import groupby

from laia.losses.ctc_loss import transform_output


class CTCGreedyDecoder:
    def __init__(self):
        self._output = None
        self._segmentation = None

    def __call__(self, x, segmentation=False):
        x, xs = transform_output(x)
        idx = x.argmax(dim=2).t().tolist()
        x = [idx_n[: xs[n].item()] for n, idx_n in enumerate(idx)]
        if segmentation:
            self._segmentation = [
                [n for n, v in enumerate(x_n) if n == 0 or v != x_n[n - 1] != 0]
                + [len(x_n) - 1]
                for x_n in x
            ]
        # Remove repeated symbols
        x = [[k for k, _ in groupby(x_n)] for x_n in x]
        # Remove CTC blank symbol
        self._output = [[x for x in x_n if x != 0] for x_n in x]
        if segmentation:
            assert any(len(self._segmentation) == len(self._output) + n for n in (1, 2))
        return self._output

    @property
    def output(self):
        return self._output

    @property
    def segmentation(self):
        return self._segmentation
