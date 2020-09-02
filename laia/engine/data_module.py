import multiprocessing
import random
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch

import laia.common.logging as log
import laia.data.transforms as transforms
from laia.data import (
    ImageFromListDataset,
    PaddingCollater,
    TextImageFromTextTableDataset,
)
from laia.data.padding_collater import by_descending_width
from laia.utils import SymbolsTable

_logger = log.get_logger(__name__)


class DataModule(pl.core.LightningDataModule):
    def __init__(
        self,
        img_dirs: List[str],
        color_mode: str,
        batch_size: int,
        tr_txt_table: Optional[str] = None,
        va_txt_table: Optional[str] = None,
        tr_shuffle: bool = True,
        tr_distortions: bool = False,
        te_img_list: Optional[List[str]] = None,
        syms: Optional[Union[Dict, SymbolsTable]] = None,
        stage: str = "fit",
    ) -> None:
        assert stage in ("fit", "test")
        base_img_transform = transforms.vision.ToImageTensor(
            mode=color_mode, invert=True
        )
        self.img_dirs = img_dirs
        self.img_channels = len(color_mode)
        self.batch_size = batch_size
        # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/2196
        self.num_workers = multiprocessing.cpu_count()
        self.pin_memory = True
        if stage == "fit":
            self.tr_ds = None
            self.va_ds = None
            self.tr_txt_table = tr_txt_table
            self.va_txt_table = va_txt_table
            self.tr_shuffle = tr_shuffle
            tr_img_transform = transforms.vision.ToImageTensor(
                mode=color_mode,
                invert=True,
                random_transform=transforms.vision.RandomBetaAffine()
                if tr_distortions
                else None,
            )
            txt_transform = transforms.text.ToTensor(syms)
            _logger.info(f"Training data transforms:\n{tr_img_transform}")
            super().__init__(
                train_transforms=(tr_img_transform, txt_transform),
                val_transforms=base_img_transform,
            )
        elif stage == "test":
            self.te_ds = None
            self.te_img_list = te_img_list
            super().__init__(test_transforms=base_img_transform)

    def setup(self, stage):
        if stage == "fit":
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
        if stage == "test":
            self.te_ds = ImageFromListDataset(
                self.te_img_list,
                img_dirs=self.img_dirs,
                img_transform=self.test_transforms,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.tr_ds is not None
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
        assert self.va_ds is not None
        return torch.utils.data.DataLoader(
            dataset=self.va_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=PaddingCollater(
                {"img": (self.img_channels, None, None)}, sort_key=by_descending_width
            ),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.te_ds is not None
        return torch.utils.data.DataLoader(
            dataset=self.te_ds,
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
