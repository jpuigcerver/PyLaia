import os
from glob import glob
from importlib import import_module
from io import BytesIO
from typing import Optional, Callable, Any, Union

import natsort as ns
import torch

from laia.common.logging import get_logger
from laia.common.random import set_rng_state

_logger = get_logger(__name__)


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


class CheckpointLoader(Loader):
    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        self._device = device
        self._loader = BasicLoader()

    def load(self, filepath: str) -> Any:
        state = self._loader.load(filepath, device=self._device)
        if state is not None:
            _logger.info("Loaded checkpoint {}", filepath)
        return state

    def load_by(
        self, pattern: str, key: Optional[Callable] = None, reverse: bool = True
    ) -> Any:
        matches = glob(pattern)
        if not len(matches):
            return None
        filepath = ns.natsorted(matches, key=key, reverse=reverse, alg=ns.ns.PATH)[0]
        return self.load(filepath)


class ModelCheckpointLoader(CheckpointLoader):
    def __init__(
        self, model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None
    ) -> None:
        super().__init__(device=device)
        self._model = model

    def load(self, filepath: str) -> Any:
        state = super().load(filepath)
        if state is not None:
            self._model.load_state_dict(state)

    def load_by(
        self, pattern: str, key: Optional[Callable] = None, reverse: bool = True
    ) -> Any:
        state = super().load_by(pattern, key=key, reverse=reverse)
        if state is not None:
            self._model.load_state_dict(state)


class StateCheckpointLoader(CheckpointLoader):
    def __init__(
        self, obj: Any, device: Optional[Union[str, torch.device]] = None
    ) -> None:
        super().__init__(device=device)
        self._obj = obj

    def load(self, filepath: str) -> Any:
        state = super().load(filepath)
        if state is not None:
            set_rng_state(state.pop("rng"), self._device)
            self._obj.load_state_dict(state)

    def load_by(
        self, pattern: str, key: Optional[Callable] = None, reverse: bool = True
    ) -> Any:
        state = super().load_by(pattern, key=key, reverse=reverse)
        if state is not None:
            set_rng_state(state.pop("rng"))
            self._obj.load_state_dict(state)
