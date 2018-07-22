from __future__ import absolute_import

import os
from glob import glob
from importlib import import_module
from io import BytesIO
from typing import Optional, Callable, Any, Union

import torch

from laia.common.logging import get_logger
from laia.common.random import set_rng_state

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

_logger = get_logger(__name__)


class Loader(object):
    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    def load(self, *args, **kwargs):
        raise NotImplementedError


class BasicLoader(Loader):
    def load(
        self,
        f,  # type: Union[str, BytesIO]
        device=None,  # type: Optional[Union[str, torch.Device]]
    ):
        # type: (...) -> Any
        try:
            # TODO: map_location accepts torch.Device in v0.4.1
            return torch.load(f, map_location=str(device) if device else None)
        except FileNotFoundError:
            _logger.info("Could not find the file {}", f)
        return None


class ObjectLoader(Loader):
    def __init__(
        self,
        f,  # type: Union[str, BytesIO]
        device=None,  # type: Optional[Union[str, torch.Device]]
    ):
        # type: (...) -> None
        self._f = f
        self._device = device
        self._loader = BasicLoader()

    def load(self):
        # type: () -> Optional
        obj = self._loader.load(self._f, device=self._device)
        if obj is None:
            return None
        module = import_module(obj["module"])
        fn = getattr(module, obj["name"])
        args = obj.get("args", [])
        kwargs = obj.get("kwargs", {})
        return fn(*args, **kwargs)


class ModelLoader(ObjectLoader):
    def __init__(self, load_path, filename="model", device=None):
        # type: (str, str, Optional[Union[str, torch.Device]]) -> None
        self._path = os.path.join(load_path, filename)
        super(ModelLoader, self).__init__(self._path, device=device)

    def load(self):
        # type: () -> Optional
        model = super(ModelLoader, self).load()
        if model is not None:
            _logger.info("Loaded model {}", self._path)
        return model


class CheckpointLoader(Loader):
    def __init__(self, device=None):
        # type: (Optional[Union[str, torch.Device]]) -> None
        self._device = device
        self._loader = BasicLoader()

    def load(self, filepath):
        # type: (str) -> Optional
        state = self._loader.load(filepath, device=self._device)
        if state is not None:
            _logger.info("Loaded checkpoint {}", filepath)
        return state

    def load_by(self, pattern, key=None, reverse=True):
        # type: (str, Optional[Callable], bool) -> Optional
        matches = glob(pattern)
        if not len(matches):
            return None
        filepath = sorted(matches, key=key, reverse=reverse)[0]
        return self.load(filepath)


class ModelCheckpointLoader(CheckpointLoader):
    def __init__(self, model, device=None):
        # type: (torch.nn.Module, Optional[Union[str, torch.Device]]) -> None
        super(ModelCheckpointLoader, self).__init__(device=device)
        self._model = model

    def load(self, filepath):
        # type: (str) -> Optional
        state = super(ModelCheckpointLoader, self).load(filepath)
        if state is not None:
            self._model.load_state_dict(state)

    def load_by(self, pattern, key=None, reverse=True):
        # type: (str, Optional[Callable], bool) -> Optional
        state = super(ModelCheckpointLoader, self).load_by(
            pattern, key=key, reverse=reverse
        )
        if state is not None:
            self._model.load_state_dict(state)


class StateCheckpointLoader(CheckpointLoader):
    def __init__(self, obj, device=None):
        # type: (Any, Optional[Union[str, torch.Device]]) -> None
        super(StateCheckpointLoader, self).__init__(device=device)
        self._obj = obj

    def load(self, filepath):
        # type: (str) -> Optional
        state = super(StateCheckpointLoader, self).load(filepath)
        if state is not None:
            set_rng_state(state.pop("rng"), self._device)
            self._obj.load_state_dict(state)

    def load_by(self, pattern, key=None, reverse=True):
        # type: (str, Optional[Callable], bool) -> Optional
        state = super(StateCheckpointLoader, self).load_by(
            pattern, key=key, reverse=reverse
        )
        if state is not None:
            set_rng_state(state.pop("rng"))
            self._obj.load_state_dict(state)
