import argparse
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

import laia.common.logging as log
from laia.common.types import Loss as LossT
from laia.engine.engine_exception import EngineException
from laia.losses.loss import Loss
from laia.utils import check_inf, check_nan


class EngineModule(pl.core.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        criterion: Callable,
        monitor: str = "va_loss",
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__()
        # forward()
        self.model = model
        # configure_optimizers()
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_l2_penalty = args.weight_l2_penalty
        self.nesterov = args.nesterov
        self.scheduler = args.scheduler
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.scheduler_monitor = args.scheduler_monitor
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
        # result_kwargs()
        self.monitor = monitor

    def configure_optimizers(self) -> Dict:
        x = {}
        if self.optimizer == "SGD":
            x["optimizer"] = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_l2_penalty,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "RMSProp":
            x["optimizer"] = torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_l2_penalty,
                momentum=self.momentum,
            )
        elif self.optimizer == "Adam":
            x["optimizer"] = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_l2_penalty,
            )
        elif self.optimizer == "AdamW":
            if self.weight_l2_penalty == 0.0:
                log.warning("Using 0.0 weight decay with AdamW")
            x["optimizer"] = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_l2_penalty,
            )
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                x["optimizer"],
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
            x["lr_dict"] = {
                "scheduler": scheduler,
                "reduce_on_plateau": True,
                "monitor": self.scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
            }
        return x

    def prepare_batch(self, batch: Any) -> Tuple[Any, Any]:
        batch_x = self.batch_input_fn(batch) if self.batch_input_fn else batch
        batch_y = self.batch_target_fn(batch) if self.batch_target_fn else None
        return batch_x, batch_y

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

    def compute_loss(self, batch: Any, batch_y_hat: Any, batch_y: Any) -> LossT:
        with self.exception_catcher(batch):
            kwargs = {}
            if isinstance(self.criterion, Loss) and self.batch_id_fn:
                kwargs = {"batch_ids": self.batch_id_fn(batch)}
            batch_loss = self.criterion(batch_y_hat, batch_y, **kwargs)
            if batch_loss is not None:
                if torch.sum(torch.isnan(batch_loss)).item() > 0:
                    raise ValueError("The loss is NaN")
                if torch.sum(torch.isinf(batch_loss)).item() > 0:
                    raise ValueError("The loss is +/- Inf")
            return batch_loss

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        with self.exception_catcher(self.current_batch):
            loss.backward()

    def training_step(self, batch: Any, batch_idx: int) -> pl.TrainResult:
        self.current_batch = batch
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher(batch):
            self.batch_y_hat = self.model(batch_x)
        self.run_checks(batch, self.batch_y_hat)
        batch_loss = self.compute_loss(batch, self.batch_y_hat, batch_y)
        result = pl.TrainResult(minimize=batch_loss)
        result.log(
            "tr_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch: Any, batch_idx: int) -> pl.EvalResult:
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher(batch):
            self.batch_y_hat = self.model(batch_x)
        self.run_checks(batch, self.batch_y_hat)
        batch_loss = self.compute_loss(batch, self.batch_y_hat, batch_y)
        result = pl.EvalResult(early_stop_on=batch_loss, checkpoint_on=batch_loss)
        result.log(
            "va_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return result

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items
