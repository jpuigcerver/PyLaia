import torch

from laia.data import PaddedTensor
from laia.nn.image_to_sequence import image_to_sequence


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
        super().__init__()
        self._horizontal = horizontal
        self._adaptive_size = adaptive_size
        self._linear = torch.nn.Linear(
            adaptive_size[0 if horizontal else 1], num_output_labels
        )

    def forward(self, x):
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        batch_size = x.size(0)

        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=self._adaptive_size)
        x = image_to_sequence(x, columnwise=self._horizontal)
        x = self._linear(x)
        xs = torch.full(
            [batch_size],
            self._adaptive_size[1 if self._horizontal else 0],
            dtype=torch.int,
        )
        return torch.nn.utils.rnn.pack_padded_sequence(
            input=x, lengths=xs, batch_first=False
        )

    def get_min_valid_image_size(self, _):
        return 1
