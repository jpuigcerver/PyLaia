import os
from glob import glob
from importlib import import_module
from io import BytesIO
from typing import Any, Callable, Optional, Union

import natsort as ns
import torch

from laia.common.logging import get_logger

_logger = get_logger(__name__)


# TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/1395
def choose_by(
    pattern: str, key: Optional[Callable] = None, reverse: bool = True
) -> Any:
    matches = glob(pattern)
    if not len(matches):
        return None
    return ns.natsorted(matches, key=key, reverse=reverse, alg=ns.ns.PATH)[0]


class Loader:
    def __call__(self, *args: Any, **kwargs: Any):
        return self.load(*args, **kwargs)

    def load(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class BasicLoader(Loader):
    def load(
        self, f: Union[str, BytesIO], device: Optional[Union[str, torch.device]] = None
    ) -> Any:
        try:
            return torch.load(f, map_location=device)
        except FileNotFoundError:
            _logger.info("Could not find the file {}", f)
        return None


class ObjectLoader(Loader):
    def __init__(
        self, f: Union[str, BytesIO], device: Optional[Union[str, torch.device]] = None
    ) -> None:
        self._f = f
        self._device = device
        self._loader = BasicLoader()

    def load(self) -> Any:
        obj = self._loader.load(self._f, device=self._device)
        if obj is None:
            return None
        module = import_module(obj["module"])
        fn = getattr(module, obj["name"])
        args = obj.get("args", [])
        kwargs = obj.get("kwargs", {})
        return fn(*args, **kwargs)


class ModelLoader(ObjectLoader):
    def __init__(
        self,
        load_path: str,
        filename: str = "model",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._path = os.path.join(load_path, filename)
        super().__init__(self._path, device=device)

    def load(self) -> Any:
        model = super().load()
        if model is not None:
            _logger.info("Loaded model {}", self._path)
        return model
