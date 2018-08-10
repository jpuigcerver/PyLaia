from typing import Union, Optional, Callable

import torch

from laia.data import PaddedTensor
from laia.engine.feeders.tensor_feeder import TensorFeeder


class ImageFeeder(TensorFeeder):
    """Feed an image as a PyTorch Tensor to the model.

    Args:
      device: Device Where the data should be allocated.
      keep_padded_tensors: Whether or not keep the size
          information of the padding. If False, the batch tensor will be
          returned without any size information. (default: True)
      keep_channels_in_size: Whether or not the number of channels of the
          images is kept as part of the size in the `PaddedTensor` objects.
      requires_grad: Whether or not the `Tensor` requires grads. (default: False)
      parent_feeder: Parent feeder that should feed this. (default: None)
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        keep_padded_tensors: bool = True,
        keep_channels_in_size: bool = False,
        requires_grad: bool = False,
        parent_feeder: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            device=device, requires_grad=requires_grad, parent_feeder=parent_feeder
        )
        self._keep_padded_tensors = keep_padded_tensors
        self._keep_channels_in_size = keep_channels_in_size

    @classmethod
    def _view_as_4d(cls, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 2:
            batch = batch.view(1, 1, batch.size(0), batch.size(1))
        elif batch.dim() == 3:
            batch = batch.view(1, batch.size(0), batch.size(1), batch.size(2))
        elif batch.dim() == 4:
            pass
        else:
            raise ValueError(
                "Tensor with {} dimensions is not supported "
                "as an Image".format(batch.dim())
            )
        return batch

    def _feed(self, batch):
        batch = super()._feed(batch)
        # View image batch as a N-C-H-W
        batch = self._view_as_4d(
            batch.data if isinstance(batch, PaddedTensor) else batch
        )
        batch.requires_grad_(self._requires_grad)
        if isinstance(batch, PaddedTensor) and self._keep_padded_tensors:
            xs = batch.sizes
            # Ensure that the size tensor is the expected
            if xs.dim() != 2 or (xs.size(1) != 2 and xs.size(1) != 3):
                raise ValueError(
                    "Size tensor in PaddedTensor has not an "
                    "expected shape: {!r}".format(xs.size())
                )
            if xs.size(1) == 3 and not self._keep_channels_in_size:
                xs = xs[:, 1:]
            return PaddedTensor(batch, xs)
        return batch
