from __future__ import absolute_import

from functools import reduce

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class CTCDecoder(object):
    def __init__(self):
        self._output = None

    def __call__(self, x):
        # Shape x: T x N x D
        if isinstance(x, PackedSequence):
            x, xs = pad_packed_sequence(x)
        elif torch.is_tensor(x):
            xs = [x.size()[0]] * x.size()[1]
        else:
            raise NotImplementedError('Not implemented for type %s' % type(x))
        if isinstance(x, Variable):
            x = x.data

        _, idx = x.max(dim=2)
        idx = idx.t().tolist()
        x = [idx_n[:int(xs[n])] for n, idx_n in enumerate(idx)]
        x = [reduce(lambda z, x: z if z[-1] == x else z + [x],
                    x_n[1:], [x_n[0]]) for x_n in x]
        self._output = [[x for x in x_n if x != 0] for x_n in x]
        return self._output

    @property
    def output(self):
        return self._output
