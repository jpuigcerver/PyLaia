import argparse
import multiprocessing
import random
from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch

import laia.common.logging as log
import laia.data.transforms as transforms
from laia.data import PaddingCollater, TextImageFromTextTableDataset
from laia.data.padding_collater import by_descending_width
from laia.utils import SymbolsTable


class DataModule(pl.core.LightningDataModule):
    def __init__(
        self, args: argparse.Namespace, syms: Union[Dict, SymbolsTable]
    ) -> None:
        tr_img_transform = transforms.vision.ToImageTensor(
            mode=args.color_mode,
            invert=True,
            random_transform=transforms.vision.RandomBetaAffine()
            if args.use_distortions
            else None,
        )
        va_img_transform = transforms.vision.ToImageTensor(
            mode=args.color_mode, invert=True
        )
        txt_transform = transforms.text.ToTensor(syms)
        log.info(f"Training data transforms:\n{tr_img_transform}")
        super().__init__(
            train_transforms=(tr_img_transform, txt_transform),
            val_transforms=va_img_transform,
        )
        self.tr_txt_table = args.tr_txt_table
        self.va_txt_table = args.va_txt_table
        self.img_dirs = args.img_dirs
        self.img_channels = len(args.color_mode)
        self.batch_size = args.batch_size
        # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/2196
        self.num_workers = multiprocessing.cpu_count()
        # TODO: if GPU
        self.pin_memory = True
        self.tr_shuffle = not bool(args.lightning.limit_train_batches)
        self.tr_ds = None
        self.va_ds = None

    def setup(self, stage=None):
        tr_img_transform, txt_transform = self.train_transforms
        self.tr_ds = TextImageFromTextTableDataset(
            self.tr_txt_table,
            self.img_dirs,
            img_transform=tr_img_transform,
            txt_transform=txt_transform,
        )
        self.va_ds = TextImageFromTextTableDataset(
            self.va_txt_table,
            self.img_dirs,
            img_transform=self.val_transforms,
            txt_transform=txt_transform,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.tr_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.tr_shuffle,
            worker_init_fn=DataModule.worker_init_fn,
            pin_memory=self.pin_memory,
            collate_fn=PaddingCollater(
                {"img": (self.img_channels, None, None)}, sort_key=by_descending_width
            ),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.va_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=PaddingCollater(
                {"img": (self.img_channels, None, None)}, sort_key=by_descending_width
            ),
        )

    @staticmethod
    def worker_init_fn(worker_id):
        # We need to reset the Numpy and Python PRNG, or we will get the
        # same numbers in each epoch (when the workers are re-generated)
        seed = (torch.initial_seed() + worker_id) % 2 ** 32  # [0, 2**32)
        random.seed(seed)
        np.random.seed(seed)
