import pytorch_lightning as pl
import torch

from laia.dummies.dummy_model import DummyModel
from laia.losses import CTCLoss


class DummyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 10 output labels: MNIST classes
        self.model = DummyModel((3, 3), 10, horizontal=True)
        self.criterion = CTCLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_y_hat = self.model(batch_x)
        loss = self.criterion(batch_y_hat, batch_y)
        self.log("tr_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "foo",
            torch.tensor(self.current_epoch),
            on_step=False,
            on_epoch=True,
            reduce_fx=max,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.log(
            "bar",
            torch.tensor(self.global_step),
            on_step=False,
            on_epoch=True,
            reduce_fx=max,
        )
