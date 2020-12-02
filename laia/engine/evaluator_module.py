from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch

from laia.engine.engine_exception import exception_catcher


class EvaluatorModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        batch_input_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        self.batch_input_fn = batch_input_fn
        self.batch_id_fn = batch_id_fn

    def test_step(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        batch_x = self.batch_input_fn(batch)
        with exception_catcher(
            self.batch_id_fn(batch) if self.batch_id_fn else batch,
            self.current_epoch,
            self.global_step,
        ):
            return self.model(batch_x)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items
