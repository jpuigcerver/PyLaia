from typing import Any, Callable, Iterator, Optional, Tuple

import pytorch_lightning as pl
import torch

from laia.common.arguments import OptimizerArgs, SchedulerArgs
from laia.common.types import Loss as LossT
from laia.engine.engine_exception import exception_catcher
from laia.losses.loss import Loss
from laia.utils import check_tensor


class EngineModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable,
        optimizer: OptimizerArgs = OptimizerArgs(),
        scheduler: Optional[SchedulerArgs] = None,
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        # configure_optimizers()
        self.optimizer = optimizer
        self.scheduler = scheduler
        # compute_loss()
        self.criterion = criterion
        # prepare_batch()
        self.batch_input_fn = batch_input_fn
        self.batch_target_fn = batch_target_fn
        # exception_catcher(), check_tensor(), compute_loss()
        self.batch_id_fn = batch_id_fn
        # training_step(), validation_step()
        self.batch_y_hat = None
        # required by auto_lr_find
        self.lr = optimizer.learning_rate

    def configure_optimizers(self):
        weight_decay = self.optimizer.weight_l2_penalty
        if self.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.optimizer.momentum,
                weight_decay=weight_decay,
                nesterov=self.optimizer.nesterov,
            )
        elif self.optimizer.name == "RMSProp":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
                momentum=self.optimizer.momentum,
            )
        elif self.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer: {self.optimizer.name}")
        if self.scheduler is not None and self.scheduler.active:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler.factor,
                    patience=self.scheduler.patience - 1,
                ),
                "monitor": self.scheduler.monitor,
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
            return [optimizer], [scheduler]
        return optimizer

    def prepare_batch(self, batch: Any) -> Tuple[Any, Any]:
        if self.batch_input_fn and self.batch_target_fn:
            return self.batch_input_fn(batch), self.batch_target_fn(batch)
        if isinstance(batch, tuple) and len(batch) == 2:
            return batch
        return batch, None

    def check_tensor(self, batch: Any, batch_y_hat: Any) -> None:
        # note: only active when logging level <= DEBUG
        check_tensor(
            tensor=batch_y_hat.data,
            logger=__name__,
            msg=(
                "Found {abs_num} ({rel_num:.2%}) infinite values in the model output "
                "at epoch={epoch}, batch={batch}, global_step={global_step}"
            ),
            epoch=self.current_epoch,
            batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
            global_step=self.global_step,
        )

    def exception_catcher(self, batch: Any) -> Iterator[None]:
        return exception_catcher(
            self.batch_id_fn(batch) if self.batch_id_fn else batch,
            self.current_epoch,
            self.global_step,
        )

    def compute_loss(self, batch: Any, batch_y_hat: Any, batch_y: Any) -> LossT:
        with self.exception_catcher(batch):
            kwargs = {}
            if isinstance(self.criterion, Loss) and self.batch_id_fn:
                kwargs = {"batch_ids": self.batch_id_fn(batch)}
            batch_loss = self.criterion(batch_y_hat, batch_y, **kwargs)
            if batch_loss is not None:
                if not torch.isfinite(batch_loss).all():
                    raise ValueError("The loss is NaN or ± inf")
            return batch_loss

    # TODO: find out how to get the batch here
    # TODO: check gradients after backward
    # def backward(self, loss, *args, **kwargs):
    #     with self.exception_catcher(batch):
    #         super().backward(loss, *args, **kwargs)

    def training_step(self, batch: Any, *_, **__):
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher(batch):
            batch_y_hat = self.model(batch_x)
        self.check_tensor(batch, batch_y_hat)
        batch_loss = self.compute_loss(batch, batch_y_hat, batch_y)
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
        return {"loss": batch_loss, "batch_y_hat": batch_y_hat}

    def validation_step(self, batch: Any, *_, **__):
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher(batch):
            batch_y_hat = self.model(batch_x)
        self.check_tensor(batch, batch_y_hat)
        batch_loss = self.compute_loss(batch, batch_y_hat, batch_y)
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
        return {"loss": batch_loss, "batch_y_hat": batch_y_hat}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items
