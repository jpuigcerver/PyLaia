from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch

from laia.losses.ctc_loss import transform_batch
from laia.utils import ArchiveLatticeWriter, ArchiveMatrixWriter


class Netout(pl.Callback):
    def __init__(
        self,
        writers: List[Union[ArchiveMatrixWriter, ArchiveLatticeWriter]],
        output_transform: Optional[str] = None,
        batch_transform: Callable = transform_batch,
    ):
        super().__init__()
        self.writers = writers
        self.output_transform = (
            getattr(torch.nn.functional, output_transform) if output_transform else None
        )
        self.batch_transform = batch_transform

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, *args, **kwargs)
        x, xs = self.batch_transform(outputs)
        x = x.detach()
        x = x.permute(1, 0, 2)
        if self.output_transform is not None:
            x = self.output_transform(x, dim=-1)
        x = [x[i, : xs[i], :] for i in range(len(xs))]
        ids = pl_module.batch_id_fn(batch)
        for writer in self.writers:
            writer.write_iterable(zip(ids, x))
