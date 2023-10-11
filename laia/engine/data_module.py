import multiprocessing
import random
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

import laia.common.logging as log
import laia.data.transforms as transforms
from laia.data import (
    ImageFromListDataset,
    PaddingCollater,
    TextImageFromTextTableDataset,
)
from laia.data.padding_collater import by_descending_width
from laia.data.unpadded_distributed_sampler import UnpaddedDistributedSampler
from laia.utils import SymbolsTable

_logger = log.get_logger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        syms: Optional[Union[Dict, SymbolsTable]] = None,
        img_dirs: Optional[List[str]] = None,
        tr_txt_table: Optional[str] = None,
        va_txt_table: Optional[str] = None,
        te_img_list: Optional[Union[str, List[str]]] = None,
        batch_size: int = 8,
        min_valid_size: Optional[int] = None,
        color_mode: str = "L",
        shuffle_tr: bool = True,
        augment_tr: bool = False,
        stage: str = "fit",
        num_workers: Optional[int] = None,
    ) -> None:
        assert stage in ("fit", "test")
        base_img_transform = transforms.vision.ToImageTensor(
            mode=color_mode, invert=True, min_width=min_valid_size
        )
        self.img_dirs = img_dirs
        self.img_channels = len(color_mode)
        self.batch_size = batch_size
        # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/2196
        self.num_workers = num_workers or multiprocessing.cpu_count()
        if stage == "fit":
            self.tr_ds = None
            self.va_ds = None
            self.tr_txt_table = tr_txt_table
            self.va_txt_table = va_txt_table
            self.shuffle_tr = shuffle_tr
            tr_img_transform = transforms.vision.ToImageTensor(
                mode=color_mode,
                invert=True,
                min_width=min_valid_size,
                random_transform=transforms.vision.RandomBetaAffine()
                if augment_tr
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

    def setup(self, stage: Optional[str] = None):
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
        elif stage == "test":
            self.te_ds = ImageFromListDataset(
                self.te_img_list,
                img_dirs=self.img_dirs,
                img_transform=self.test_transforms,
            )
        else:
            raise ValueError

    def get_unpadded_distributed_sampler(
        self, ds: torch.utils.data.Dataset
    ) -> Optional[DistributedSampler]:
        if not self.trainer.use_ddp:
            return
        return UnpaddedDistributedSampler(
            ds,
            num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
            rank=self.trainer.global_rank,
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.tr_ds is not None
        return DataLoader(
            dataset=self.tr_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_tr,
            worker_init_fn=DataModule.worker_init_fn,
            pin_memory=self.trainer.on_gpu,
            collate_fn=PaddingCollater(
                {"img": (self.img_channels, None, None)}, sort_key=by_descending_width
            ),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.va_ds is not None
        return DataLoader(
            dataset=self.va_ds,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.get_unpadded_distributed_sampler(self.va_ds),
            num_workers=self.num_workers,
            pin_memory=self.trainer.on_gpu,
            collate_fn=PaddingCollater(
                {"img": (self.img_channels, None, None)}, sort_key=by_descending_width
            ),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.te_ds is not None
        return DataLoader(
            dataset=self.te_ds,
            batch_size=self.batch_size,
            sampler=self.get_unpadded_distributed_sampler(self.te_ds),
            num_workers=self.num_workers,
            pin_memory=self.trainer.on_gpu,
            collate_fn=PaddingCollater(
                {"img": (self.img_channels, None, None)}, sort_key=by_descending_width
            ),
        )

    @staticmethod
    def worker_init_fn(worker_id):
        # We need to reset the Numpy and Python PRNG, or we will get the
        # same numbers in each epoch (when the workers are re-generated)
        seed = (torch.initial_seed() + worker_id) % 2**32  # [0, 2**32)
        random.seed(seed)
        np.random.seed(seed)
