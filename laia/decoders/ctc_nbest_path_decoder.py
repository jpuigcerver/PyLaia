from __future__ import absolute_import

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class CTCNBestPathDecoder(object):
    """N-best path decoder based on CTC output.

    Examples:
        >>> nbest_decoder = CTCNBestPathDecoder(4)
        >>> a = torch.Tensor([[[ 1.0, 3.0, -1.0, 0.0]], \
                              [[-1.0, 2.0, -2.0, 3.0]], \
                              [[ 1.0, 5.0,  9.0, 2.0]], \
                              [[-1.0,-2.0, -3.0,-4.0]]])
        >>> nbest_decoder(a)
        [[(14.0, [1, 3, 2, 0]), (13.0, [1, 3, 2, 1]), (13.0, [1, 1, 2, 0]), (12.0, [1, 3, 2, 2])]]
    """
    def __init__(self, nbest):
        assert isinstance(nbest, int) and nbest > 0
        self._nbest = nbest
        self._output = None

    def __call__(self, x):
        # Shape x: T x N x D
        if isinstance(x, PackedSequence):
            x, xs = pad_packed_sequence(x)
        elif torch.is_tensor(x):
            xs = [x.size()[0]] * x.size()[1]
        else:
            raise NotImplementedError('Not implemented for type %s' % type(x))

        x = x.permute(1, 0, 2)
        x = x.cpu()

        self._output = []
        for matrix, length in zip(x, xs):
            topk_val, topk_idx = torch.topk(matrix,
                                            k=min(self._nbest, matrix.size(1)),
                                            dim=1)
            best_paths = [(float(v), [int(i)])
                          for v, i in zip(topk_val[0], topk_idx[0])]

            for t in range(1, length):
                best_paths = [(lkh + float(v), path + [int(i)])
                              for (lkh, path) in best_paths
                              for v, i in zip(topk_val[t], topk_idx[t])]
                best_paths.sort(reverse=True)
                best_paths = best_paths[:self._nbest]

            self._output.append(best_paths)

        return self._output

    @property
    def output(self):
        return self._output
