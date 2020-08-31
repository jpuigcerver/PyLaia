from contextlib import contextmanager
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch

from laia.engine.engine_exception import EngineException


class EvaluatorModule(pl.core.LightningModule):
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
        self.batch_y_hat = None

    @contextmanager
    def exception_catcher(self, batch: Any) -> Any:
        try:
            yield
        except Exception as e:
            raise EngineException(
                epoch=self.current_epoch,
                global_step=self.global_step,
                batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
                cause=e,
            ) from e

    def test_step(self, batch: Any, batch_idx: int):
        super().test_step(batch, batch_idx)
        batch_x = self.batch_input_fn(batch)
        with self.exception_catcher(batch):
            self.batch_y_hat = self.model(batch_x)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        return None
