from __future__ import absolute_import

from functools import reduce

import torch

from laia.losses.ctc_loss import transform_output

"""
from laia.decoders.ctc_lattice_generator import CTCLatticeGenerator

import pywrapfst as fst
"""


class CTCNBestPathDecoder(object):
    """N-best path decoder based on CTC output.

    Examples:
        >>> a = torch.tensor([[[ 1.0, 3.0, -1.0, 0.0]], \
                              [[-1.0, 2.0, -2.0, 3.0]], \
                              [[ 1.0, 5.0,  9.0, 2.0]], \
                              [[-1.0,-2.0, -3.0,-4.0]]])
        >>> nbest_decoder = CTCNBestPathDecoder(4, output_alignment=False)
        >>> nbest_decoder(a)
        [[(14.0, [1, 3, 2]), (13.0, [1, 3, 2, 1]), (13.0, [1, 2]), (12.0, [1, 3, 2])]]
        >>> nbest_decoder = CTCNBestPathDecoder(4, output_alignment=True)
        >>> nbest_decoder(a)
        [[(14.0, [1, 3, 2, 0]), (13.0, [1, 3, 2, 1]), (13.0, [1, 1, 2, 0]), (12.0, [1, 3, 2, 2])]]
    """

    def __init__(self, nbest, output_alignment=False):
        assert isinstance(nbest, int) and nbest > 0
        self._nbest = nbest
        self._output_alignment = output_alignment
        self._output = None

    def __call__(self, x):
        """
        latgen = CTCLatticeGenerator(normalize=True)
        lattices = latgen(x)
        self._output = [fst.shortestpath(f, nshortest=self._nbest)
                        for f in lattices]
        return self._output
        """
        x, xs = transform_output(x)
        x = x.permute(1, 0, 2)
        x = x.cpu()

        self._output = []
        for matrix, length in zip(x, xs):
            topk_val, topk_idx = torch.topk(
                matrix, k=min(self._nbest, matrix.size(1)), dim=1
            )
            best_paths = [
                (float(v), [int(i)]) for v, i in zip(topk_val[0], topk_idx[0])
            ]

            for t in range(1, length):
                best_paths = [
                    (lkh + float(v), path + [int(i)])
                    for (lkh, path) in best_paths
                    for v, i in zip(topk_val[t], topk_idx[t])
                ]
                best_paths.sort(reverse=True)
                best_paths = best_paths[: self._nbest]

            if not self._output_alignment:
                best_paths = [
                    (
                        lkh,
                        reduce(
                            lambda z, x: z if z[-1] == x else z + [x],
                            path[1:],
                            [path[0]],
                        ),
                    )
                    for lkh, path in best_paths
                ]
                best_paths = [
                    (lkh, [x for x in path if x != 0]) for lkh, path in best_paths
                ]

            self._output.append(best_paths)

        return self._output

    @property
    def output(self):
        return self._output
