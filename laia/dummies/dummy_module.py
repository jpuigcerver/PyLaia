from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision

from laia.data.transforms.vision import ToImageTensor
from laia.dummies.dummy_model import DummyModel
from laia.losses import CTCLoss


class DummyModule(pl.LightningModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.model = DummyModel((3, 3), 10, horizontal=True)
        self.criterion = CTCLoss()
        self.root = Path(__file__).parent.parent.parent / "datasets"

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def collate_fn(self, batch):
        x = torch.stack([a for a, b in batch])
        y = [[b] for a, b in batch]
        return x, y

    def train_dataloader(self):
        mnist = torchvision.datasets.MNIST(
            self.root, train=True, download=True, transform=ToImageTensor()
        )
        return torch.utils.data.DataLoader(
            mnist, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn
        )

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_y_hat = self.model(batch_x)
        loss = self.criterion(batch_y_hat, batch_y)
        result = pl.TrainResult(minimize=loss)
        result.log("tr_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        result.log(
            "foo", torch.tensor(self.current_epoch, dtype=torch.float), reduce_fx=max
        )
        result.log(
            "bar",
            torch.tensor(self.global_step, dtype=torch.float),
            on_step=False,
            on_epoch=True,
            reduce_fx=max,
        )
        return result
