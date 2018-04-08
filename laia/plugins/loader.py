from __future__ import absolute_import

import io
import json
from importlib import import_module

import os.path as p
import torch

from laia.logging import get_logger
from laia.random import set_rng_state

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

_logger = get_logger(__name__)


class Loader(object):
    def __init__(self, load_path, filename):
        assert not p.dirname(filename)
        assert p.exists(load_path)
        self._load_path = p.normpath(load_path)
        self._filename = filename
        self._path = p.join(load_path, filename)

    @property
    def path(self):
        return self._path

    @staticmethod
    def load_binary(path):
        try:
            return torch.load(path)
        except FileNotFoundError:
            _logger.info('Could not find the file {}', path)
            return None


class ObjectLoader(Loader):
    def __call__(self):
        return self.load()

    def load(self):
        obj = self.load_binary(self.path)
        if obj is None:
            return None
        module = import_module(obj['module'])
        fn = getattr(module, obj['name'])
        args = obj.get('args', [])
        kwargs = obj.get('kwargs', {})
        return fn(*args, **kwargs)


class ModelLoader(ObjectLoader):
    def __init__(self, load_path, filename='model'):
        super(ModelLoader, self).__init__(load_path, filename)

    def load(self):
        try:
            model = super(ModelLoader, self).load()
            if model is not None:
                _logger.info('Loaded model {}', self.path)
            return model
        except Exception as e:
            _logger.error('Error while loading the model {}: {}', self.path, e)


class TrainerLoader(ObjectLoader):
    def __init__(self, load_path, filename='trainer'):
        super(TrainerLoader, self).__init__(load_path, filename)

    def load(self):
        try:
            trainer = super(TrainerLoader, self).load()
            if trainer is not None:
                _logger.info('Loaded trainer {}', self.path)
            return trainer
        except Exception as e:
            _logger.error('Error while loading the trainer {}: {}', self.path, e)


class CheckpointLoader(Loader):
    def __call__(self):
        return self.load()

    def _get_last_ckpt_path(self):
        path = p.join(self._load_path, '.ledger.json')
        with io.open(path, 'r') as f:
            ledger = json.load(f)
            return ledger.get(self._filename, None)

    def _get_ckpt_path_by(self, criterion):
        path = p.join(
            self._load_path,
            '{}-{}'.format(self._filename, criterion))
        return path

    def load(self):
        path = p.join(self._load_path, self._filename)
        try:
            state = self.load_binary(path)
            if state is not None:
                _logger.info('Loaded checkpoint {}', path)
            return state
        except Exception as e:
            _logger.error('Error while loading the checkpoint {}: {}', path, e)

    def load_last(self):
        path = self._get_last_ckpt_path()
        if path is None:
            _logger.info('No previous checkpoint found')
            return None
        try:
            state = self.load_binary(path)
            if state is not None:
                _logger.info('Loaded last checkpoint {}', path)
            return state
        except Exception as e:
            _logger.error('Error while loading the checkpoint {}: {}', path, e)

    def load_by(self, criterion):
        path = self._get_ckpt_path_by(criterion)
        try:
            state = self.load_binary(path)
            if state is not None:
                _logger.info('Loaded {} checkpoint {}', criterion, path)
            return state
        except Exception as e:
            _logger.error('Error while loading the checkpoint {}: {}', path, e)


class ModelCheckpointLoader(CheckpointLoader):
    def __init__(self, load_path, name='model.ckpt'):
        super(ModelCheckpointLoader, self).__init__(load_path, name)


class TrainerCheckpointLoader(CheckpointLoader):
    def __init__(self, load_path, name='trainer.ckpt'):
        super(TrainerCheckpointLoader, self).__init__(load_path, name)

    def load(self):
        state = super(TrainerCheckpointLoader, self).load()
        if state is not None:
            set_rng_state(state.pop('rng_state'))
        return state

    def load_last(self):
        state = super(TrainerCheckpointLoader, self).load_last()
        if state is not None:
            set_rng_state(state.pop('rng_state'))
        return state

    def load_by(self, criterion):
        state = super(TrainerCheckpointLoader, self).load_by(criterion)
        if state is not None:
            set_rng_state(state.pop('rng_state'))
        return state
