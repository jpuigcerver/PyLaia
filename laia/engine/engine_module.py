from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from laia.common.types import Loss as LossT
from laia.engine.engine_exception import exception_catcher
from laia.losses.loss import Loss
from laia.utils import check_inf, check_nan


class EngineModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str,
        criterion: Callable,
        optimizer_kwargs: Optional[Dict] = None,
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        # configure_optimizers()
        self.optimizer = optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs
        # compute_loss()
        self.criterion = criterion
        # prepare_batch()
        self.batch_input_fn = batch_input_fn
        self.batch_target_fn = batch_target_fn
        # exception_catcher(), run_checks(), compute_loss()
        self.batch_id_fn = batch_id_fn
        # training_step(), validation_step()
        self.batch_y_hat = None
        # backward()
        self.current_batch = None
        # required by auto_lr_find
        self.lr = self.optimizer_kwargs.get("learning_rate", 5e-4)

    def configure_optimizers(self):
        weight_decay = self.optimizer_kwargs.get("weight_l2_penalty", 0)
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.optimizer_kwargs.get("momentum", 0),
                weight_decay=weight_decay,
                nesterov=self.optimizer_kwargs.get("nesterov", False),
            )
        elif self.optimizer == "RMSProp":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
                momentum=self.optimizer_kwargs.get("momentum", 0),
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer: {self.optimizer}")
        if self.optimizer_kwargs.get("scheduler", False):
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.optimizer_kwargs.get("scheduler_factor", 0.1),
                    patience=self.optimizer_kwargs.get("scheduler_patience", 5) - 1,
                ),
                "monitor": self.optimizer_kwargs.get("scheduler_monitor", "va_loss"),
                "interval": "epoch",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        return optimizer

    def prepare_batch(self, batch: Any) -> Tuple[Any, Any]:
        if self.batch_input_fn and self.batch_target_fn:
            return self.batch_input_fn(batch), self.batch_target_fn(batch)
        if isinstance(batch, tuple) and len(batch) == 2:
            return batch
        return batch, None

    def run_checks(self, batch: Any, batch_y_hat: Any) -> None:
        # Note: only active when logging level <= DEBUG
        check_inf(
            tensor=batch_y_hat.data,
            logger=__name__,
            msg=(
                "Found {abs_num} ({rel_num:.2%}) INF values in the model output "
                "at epoch={epoch}, batch={batch}, global_step={global_step}"
            ),
            epoch=self.current_epoch,
            batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
            global_step=self.global_step,
        )
        check_nan(
            tensor=batch_y_hat.data,
            logger=__name__,
            msg=(
                "Found {abs_num} ({rel_num:.2%}) NAN values in the model output "
                "at epoch={epoch}, batch={batch}, global_step={global_step}"
            ),
            epoch=self.current_epoch,
            batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
            global_step=self.global_step,
        )

    def exception_catcher(self):
        return exception_catcher(
            self.batch_id_fn(self.current_batch)
            if self.batch_id_fn
            else self.current_batch,
            self.current_epoch,
            self.global_step,
        )

    def compute_loss(self, batch: Any, batch_y_hat: Any, batch_y: Any) -> LossT:
        with self.exception_catcher():
            kwargs = {}
            if isinstance(self.criterion, Loss) and self.batch_id_fn:
                kwargs = {"batch_ids": self.batch_id_fn(batch)}
            batch_loss = self.criterion(batch_y_hat, batch_y, **kwargs)
            if batch_loss is not None:
                if torch.sum(torch.isnan(batch_loss)).item() > 0:
                    raise ValueError("The loss is NaN")
                if torch.sum(torch.isinf(batch_loss)).item() > 0:
                    raise ValueError("The loss is ± inf")
            return batch_loss

    def backward(self, loss, *_, **__):
        with self.exception_catcher():
            loss.backward()

    def training_step(self, batch: Any, *_, **__):
        self.current_batch = batch
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher():
            self.batch_y_hat = self.model(batch_x)
        self.run_checks(batch, self.batch_y_hat)
        batch_loss = self.compute_loss(batch, self.batch_y_hat, batch_y)
        if batch_loss is None:
            return
        self.log(
            "tr_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return batch_loss

    def validation_step(self, batch: Any, *_, **__):
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher():
            self.batch_y_hat = self.model(batch_x)
        self.run_checks(batch, self.batch_y_hat)
        batch_loss = self.compute_loss(batch, self.batch_y_hat, batch_y)
        if batch_loss is None:
            return
        self.log(
            "va_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return batch_loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items
