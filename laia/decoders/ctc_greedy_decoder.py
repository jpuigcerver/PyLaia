from __future__ import absolute_import

from functools import reduce

from torch.autograd import Variable

from laia.losses.ctc_loss import transform_output


class CTCGreedyDecoder(object):

    def __init__(self):
        self._output = None

    def __call__(self, x):
        x, xs = transform_output(x)
        if isinstance(x, Variable):
            x = x.data
        _, idx = x.max(dim=2)
        idx = idx.t().tolist()
        x = [idx_n[: int(xs[n])] for n, idx_n in enumerate(idx)]
        # Remove repeated symbols
        x = [
            reduce(lambda z, x: z if z[-1] == x else z + [x], x_n[1:], [x_n[0]])
            for x_n in x
        ]
        # Remove CTC blank symbol
        self._output = [[x for x in x_n if x != 0] for x_n in x]
        return self._output

    @property
    def output(self):
        return self._output
