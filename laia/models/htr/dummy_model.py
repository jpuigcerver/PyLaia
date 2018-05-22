from __future__ import absolute_import

import torch

from laia.data.padding_collater import PaddedTensor
from laia.nn.image_to_sequence import image_to_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import adaptive_avg_pool2d


class DummyModel(torch.nn.Module):
    """Dummy HTR model for tests

    First, this does an adaptive average pooling converting each images to a
    fixed output size of `adaptive_size`.

    Then, the fixed-size image is transformed into a sequence column/row-wise
    according to the direction of the text (`horizontal`).

    Finally, for each timestep in the sequence, a linear transformation is done
    to have `num_output_labels` at each timestep.

    Returns a PackedSequence (all samples have actually the same size).
    """

    def __init__(self, adaptive_size, num_output_labels, horizontal=True):
        super(DummyModel, self).__init__()
        self._horizontal = horizontal
        self._adaptive_size = adaptive_size
        self._linear = torch.nn.Linear(
            adaptive_size[0] if horizontal else adaptive_size[1], num_output_labels
        )

    def forward(self, x):
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        batch_size = x.size(0)

        x = adaptive_avg_pool2d(x, output_size=self._adaptive_size)
        x = image_to_sequence(x, columnwise=self._horizontal)
        x = self._linear(x)

        if self._horizontal:
            xs = torch.IntTensor(batch_size).fill_(self._adaptive_size[1])
            return pack_padded_sequence(input=x, lengths=xs.tolist(), batch_first=False)
        else:
            xs = torch.IntTensor(batch_size).fill_(self._adaptive_size[0])
            return pack_padded_sequence(input=x, lengths=xs.tolist(), batch_first=False)
