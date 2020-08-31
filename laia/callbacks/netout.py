from typing import Callable, List, Union

import pytorch_lightning as pl
import torch.nn.functional as functional

from laia.losses.ctc_loss import transform_output
from laia.utils import ArchiveLatticeWriter, ArchiveMatrixWriter


class Netout(pl.callbacks.Callback):
    def __init__(
        self,
        output_transform: str,
        writers: List[Union[ArchiveMatrixWriter, ArchiveLatticeWriter]],
        model_output_fn: Callable = transform_output,
    ):
        super().__init__()
        self.output_transform = output_transform
        self.writers = writers
        self.model_output_fn = model_output_fn

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, batch, batch_idx, dataloader_idx)
        x, xs = self.model_output_fn(pl_module.batch_y_hat)
        x = x.detach()
        x = x.permute(1, 0, 2)
        if self.output_transform:
            x = getattr(functional, self.output_transform)(x, dim=-1)
        x = [x[i, : xs[i], :] for i in range(len(xs))]
        # TODO: does this play well with ddp?
        x = [x_n.cpu().numpy() for x_n in x]
        ids = pl_module.batch_id_fn(batch)
        for id_n, x_n in zip(ids, x):
            for writer in self.writers:
                writer.write(id_n, x_n)
