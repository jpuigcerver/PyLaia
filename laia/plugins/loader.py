from __future__ import absolute_import

import os
from importlib import import_module

import torch

from laia.logging import get_logger
from laia.random import set_rng_state

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
    def load(self, filepath, gpu=None):
        device = "cuda:{}".format(gpu - 1) if gpu else "cpu"
        try:
            return torch.load(filepath, map_location=device)
        except FileNotFoundError:
            _logger.info("Could not find the file {}", filepath)
        return None


class ObjectLoader(Loader):
    def __init__(self, filepath, gpu=None):
        self._filepath = filepath
        self._gpu = gpu
        self._loader = BasicLoader()

    def load(self):
        obj = self._loader.load(self._filepath, gpu=self._gpu)
        if obj is None:
            return None
        module = import_module(obj["module"])
        fn = getattr(module, obj["name"])
        args = obj.get("args", [])
        kwargs = obj.get("kwargs", {})
        return fn(*args, **kwargs)


class ModelLoader(ObjectLoader):
    def __init__(self, load_path, filename="model", gpu=None):
        self._path = os.path.join(load_path, filename)
        super(ModelLoader, self).__init__(self._path, gpu=gpu)

    def load(self):
        model = super(ModelLoader, self).load()
        if model is not None:
            _logger.info("Loaded model {}", self._path)
        return model


class TrainerLoader(ObjectLoader):
    def __init__(self, load_path, filename="trainer", gpu=None):
        self._path = os.path.join(load_path, filename)
        super(TrainerLoader, self).__init__(self._path, gpu=gpu)

    def load(self):
        trainer = super(TrainerLoader, self).load()
        if trainer is not None:
            _logger.info("Loaded trainer {}", self._path)
        return trainer


class CheckpointLoader(Loader):
    def __init__(self, gpu=None):
        self._gpu = gpu
        self._loader = BasicLoader()

    def load(self, filepath):
        state = self._loader.load(filepath, gpu=self._gpu)
        if state is not None:
            _logger.info("Loaded checkpoint {}", filepath)
        return state

    def load_by(self, pattern, key=None, reverse=True):
        import glob

        matches = glob.glob(pattern)
        if not len(matches):
            return None
        filepath = sorted(matches, key=key, reverse=reverse)[0]
        return self.load(filepath)


class ModelCheckpointLoader(CheckpointLoader):
    def __init__(self, model, gpu=None):
        super(ModelCheckpointLoader, self).__init__(gpu=gpu)
        self._model = model

    def load(self, filepath):
        state = super(ModelCheckpointLoader, self).load(filepath)
        if state is not None:
            self._model.load_state_dict(state)

    def load_by(self, pattern, key=None, reverse=True):
        state = super(ModelCheckpointLoader, self).load_by(
            pattern, key=key, reverse=reverse
        )
        if state is not None:
            self._model.load_state_dict(state)


class TrainerCheckpointLoader(CheckpointLoader):
    def __init__(self, trainer, gpu=None):
        super(TrainerCheckpointLoader, self).__init__(gpu=gpu)
        self._trainer = trainer

    def load(self, filepath):
        state = super(TrainerCheckpointLoader, self).load(filepath)
        if state is not None:
            set_rng_state(state.pop("rng_state"), self._gpu)
            self._trainer.load_state_dict(state)

    def load_by(self, pattern, key=None, reverse=True):
        state = super(TrainerCheckpointLoader, self).load_by(
            pattern, key=key, reverse=reverse
        )
        if state is not None:
            set_rng_state(state.pop("rng_state"))
            self._trainer.load_state_dict(state)
