import shutil
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision

from laia.data import PaddingCollater, TextImageFromTextTableDataset
from laia.dummies import DummyMNIST


class DummyMNISTLines(DummyMNIST):
    def __init__(
        self,
        max_length: int = 80,
        tr_n: int = 30,
        va_n: int = 15,
        space_sym: Optional[str] = "<space>",
        samples_per_space: Optional[int] = None,
        batch_size: int = 8,
    ):
        super().__init__(batch_size=batch_size)
        self.root = self.root / "MNIST-lines"
        self.max_length = max_length
        self.n = {"tr": tr_n, "va": va_n}
        self.space_sym = space_sym
        self.samples_per_space = samples_per_space
        # prepare symbols table
        self.syms = {i: str(i) for i in range(10)}
        self.syms[10] = "<ctc>"
        if space_sym is not None and samples_per_space is not None:
            self.syms[11] = space_sym

    @staticmethod
    def get_indices(
        max_length: int,
        total_size: int,
        samples_per_space: Optional[int] = None,
    ) -> List[Union[str, int]]:
        n_samples = np.random.randint(1, high=max_length + 1)
        indices = list(np.random.choice(total_size, size=n_samples))
        if samples_per_space is not None and n_samples > samples_per_space:
            n_spaces = np.random.randint(0, high=(n_samples // samples_per_space) + 1)
            arange = np.arange(1, len(indices))  # no spaces at beginning or end
            space_indices = np.random.choice(arange, size=n_spaces, replace=False)
            n_samples += n_spaces
            for space_i in sorted(space_indices, reverse=True):
                indices.insert(space_i, "sp")
        return indices

    @staticmethod
    def concatenate(
        dataset,
        h: int,
        w: int,
        indices: List[Union[str, int]],
        invert: bool = False,
        space_sym: Optional[str] = None,
    ):
        img = np.empty((h, w * len(indices)))
        txt = []
        # mask to avoid carving space pixels
        space_mask = np.zeros_like(img)
        for i, idx in enumerate(indices):
            w_slice = slice(w * i, w * (i + 1))
            if idx == "sp":
                # add space pixels
                img[:, w_slice] = int(invert)
                txt.append(space_sym)
                space_mask[:, w_slice] = 1
            else:
                # add actual image
                x, y = dataset[idx]
                if isinstance(x, torch.Tensor):
                    x = x.numpy()
                img[:, w_slice] = x
                txt.append(str(y))
        txt = " ".join(txt)
        return img, txt, space_mask

    def prepare_data(self):
        datasets = {
            "tr": torchvision.datasets.MNIST(
                self.root.parent, train=True, download=True
            ),
            "va": torchvision.datasets.MNIST(
                self.root.parent, train=False, download=True
            ),
        }
        for k, ds in datasets.items():
            # prepare img directory
            data_dir = self.root / k
            if data_dir.exists():
                shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            # prepare ground-truth file and indices per sample file
            gt_file = open(data_dir.parent / f"{k}.gt", mode="w")
            indices_file = open(data_dir.parent / f"{k}.indices", mode="w")

            for i in range(self.n[k]):
                # get random dataset indices
                indices = self.get_indices(
                    self.max_length,
                    total_size=len(ds),
                    samples_per_space=self.samples_per_space,
                )

                # concatenate selected images into a line image
                img, txt, _ = self.concatenate(
                    ds, 28, 28, indices, space_sym=self.space_sym
                )

                # save line image samples
                img_id = f"{k}-{i}"
                torchvision.utils.save_image(
                    torch.from_numpy(img), data_dir / f"{img_id}.jpg"
                )
                # save reference text
                gt_file.write(f"{img_id} {txt}\n")
                # save indices used in each sample
                indices_file.write(f"{img_id} {[str(idx) for idx in indices]}\n")

            gt_file.close()
            indices_file.close()

    def setup(self, _):
        self.tr_ds = TextImageFromTextTableDataset(
            self.root / "tr.gt",
            self.root / "tr",
            img_transform=self.train_transforms,
        )
        self.va_ds = TextImageFromTextTableDataset(
            self.root / "va.gt",
            self.root / "va",
            img_transform=self.train_transforms,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.tr_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=PaddingCollater({"img": (1, None, None)}),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.va_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=PaddingCollater({"img": (1, None, None)}),
        )
