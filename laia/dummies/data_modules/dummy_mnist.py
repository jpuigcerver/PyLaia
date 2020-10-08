from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision

from laia.data.transforms.vision import ToImageTensor


class DummyMNIST(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.root = Path(__file__).parent.parent.parent / "datasets"
        super().__init__(
            train_transforms=ToImageTensor(),
            val_transforms=ToImageTensor(),
        )
        self.tr_ds = None
        self.va_ds = None

    def prepare_data(self):
        torchvision.datasets.MNIST(self.root, train=True, download=True)
        torchvision.datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage):
        self.tr_ds = torchvision.datasets.MNIST(
            self.root,
            train=stage == "fit",
            transform=self.train_transforms,
        )
        self.va_ds = torchvision.datasets.MNIST(
            self.root,
            train=stage != "fit",
            transform=self.val_transforms,
        )

    def collate_fn(self, batch):
        x = torch.stack([a for a, b in batch])
        y = [[b] for a, b in batch]
        return x, y

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.tr_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.va_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()
