from typing import (
    Any,
    Callable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch


class PaddedTensor(NamedTuple):
    data: torch.Tensor
    sizes: torch.Tensor

    @classmethod
    def build(cls, data, sizes):
        assert sizes.dim() == 2, "PaddedTensor.sizes must have 2 dimensions"
        assert sizes.size(1) in (2, 3), (
            "PaddedTensor.sizes is incorrect: "
            "expected=2 (HxW) or 3 (CxHxW), "
            f"found={sizes.size(1)}"
        )
        assert data.size(0) == sizes.size(0), (
            f"Batch size {sizes.size(0)} does not match "
            f"the number of samples in the batch {data.size(0)}"
        )
        return cls(data, sizes)

    def __repr__(self) -> str:
        return (
            f"PaddedTensor("
            f"data={list(self.data.size())}, "
            f"sizes={self.sizes.tolist()}, "
            f"device={str(self.data.device)})"
        )

    @property
    def device(self) -> torch.device:
        return self.data.device


def by_descending_width(x):
    # C x H x W
    return -x["img"].size(2)


class PaddingCollater:
    def __init__(self, sizes: Any, sort_key: Callable = None):
        self._sizes = sizes
        self._sort_key = sort_key

    def __call__(self, batch: Any) -> torch.Tensor:
        if self._sort_key:
            batch = sorted(batch, key=self._sort_key)
        return self.collate(batch, self._sizes)

    @staticmethod
    def get_max_sizes(
        batch: List[torch.Tensor], sizes: Optional[Tuple[Union[int, None], ...]] = None
    ) -> Tuple[int, ...]:
        # All tensors in the batch must have the same number of dimensions
        dim = batch[0].dim()
        assert all(x.dim() == dim for x in batch)
        max_sizes = [len(batch)]
        for d in range(dim):
            max_v = max(x.size(d) for x in batch)
            min_v = min(x.size(d) for x in batch)
            if sizes and sizes[d] is not None:
                assert max_v == min_v == sizes[d]
            max_sizes.append(max_v)
        return tuple(max_sizes)

    @staticmethod
    def collate_tensors(
        batch: List[torch.Tensor], max_sizes: Tuple[int, ...]
    ) -> torch.Tensor:
        out = batch[0].new_zeros(size=max_sizes)
        for i, x in enumerate(batch):
            batch_tensor = out[i]
            for d in range(batch[0].dim()):
                batch_tensor = batch_tensor.narrow(d, 0, x.size(d))
            batch_tensor.add_(x)
        return out

    def collate(self, batch: Any, sizes: Any) -> Any:
        elem, elem_type = batch[0], type(batch[0])
        if isinstance(elem, torch.Tensor):
            if any(s is None for s in sizes):
                max_sizes = PaddingCollater.get_max_sizes(batch, sizes)
                x = PaddingCollater.collate_tensors(batch, max_sizes)
                xs = torch.stack([torch.tensor(x.size()) for x in batch])
                return PaddedTensor.build(x, xs)
            else:
                return torch.stack(batch)
        elif isinstance(elem, np.ndarray):
            return self.collate([torch.from_numpy(b) for b in batch], sizes)
        elif isinstance(elem, Mapping):
            return {
                k: self.collate([d[k] for d in batch], sizes[k])
                if k in sizes
                else [d[k] for d in batch]
                for k in elem
            }
        elif isinstance(elem, Sequence):
            return [self.collate(b, s) for b, s in zip(batch, sizes)]
        raise TypeError(
            f"Batch must contain tensors, numbers, dicts or lists. Found {elem_type}"
        )
