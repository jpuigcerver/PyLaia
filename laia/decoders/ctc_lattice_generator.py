from __future__ import absolute_import

import pywrapfst as fst
from torch.nn.functional import log_softmax

from laia.losses.ctc_loss import transform_output


class CTCLatticeGenerator(object):
    def __init__(self, normalize=False):
        self._normalize = normalize

    def __call__(self, x):
        x, xs = transform_output(x)
        # Normalize log-posterior matrices, if necessary
        if self._normalize:
            x = log_softmax(x, dim=2)
        x = x.permute(1, 0, 2).cpu()
        self._output = []
        D = x.size(2)
        for logpost, length in zip(x, xs):
            f = fst.Fst()
            f.set_start(f.add_state())
            for t in range(length):
                f.add_state()
                for j in range(D):
                    weight = fst.Weight(f.weight_type(), float(-logpost[t, j]))
                    f.add_arc(
                        t,
                        fst.Arc(
                            j + 1,  # input label
                            j + 1,  # output label
                            weight,  # -logpost[t, j]
                            t + 1,  # nextstate
                        ),
                    )
            f.set_final(length, fst.Weight.One(f.weight_type()))
            f.verify()
            self._output.append(f)
        return self._output

    @property
    def output(self):
        return self._output
