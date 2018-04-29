from __future__ import absolute_import

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.nn.functional import log_softmax

import pywrapfst as fst


class CTCLatticeGenerator(object):
    def __init__(self, normalize=False):
        self._normalize = normalize

    def __call__(self, x):
        # Shape x: T x N x D
        if isinstance(x, PackedSequence):
            x, xs = pad_packed_sequence(x)
        elif torch.is_tensor(x):
            xs = [x.size()[0]] * x.size()[1]
        else:
            raise NotImplementedError('Not implemented for type %s' % type(x))

        # Normalize log-posterior matrices, if necessary
        if self._normalize:
            if not isinstance(x, torch.autograd.Variable):
                x = torch.autograd.Variable(x, requires_grad=False)
            x = log_softmax(x, dim=2)

        x = x.permute(1, 0, 2)
        x = x.cpu()

        self._output = []
        D = x.size(2)
        for logpost, length in zip(x, xs):
            f = fst.Fst()
            f.set_start(f.add_state())
            for t in range(length):
                f.add_state()
                for j in range(D):
                    weight = fst.Weight(f.weight_type(), float(-logpost[t, j]))
                    f.add_arc(t, fst.Arc(j + 1,   # input label
                                         j + 1,   # output label
                                         weight,    # -logpost[t, j]
                                         t + 1))  # nextstate
            f.set_final(length, fst.Weight.One(f.weight_type()))
            f.verify()
            self._output.append(f)
        return self._output

    @property
    def output(self):
        return self._output

