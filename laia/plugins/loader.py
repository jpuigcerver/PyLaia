from __future__ import absolute_import

import io
import json
import os
import os.path as p
from importlib import import_module

import torch

from laia.logging import get_logger
from laia.random import set_rng_state

_logger = get_logger(__name__)


class Loader(object):
    def __init__(self, save_path, filename):
        assert not p.dirname(filename)
        assert any(f.startswith(filename) for f in os.listdir(save_path))
        self._save_path = save_path
        self._filename = filename

    @staticmethod
    def load_json(path):
        with io.open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_binary(path):
        return torch.load(path)


class ModelLoader(Loader):
    def __call__(self):
        return self.load()

    def load(self):
        path = p.join(self._save_path, self._filename)
        try:
            model = self.load_json(path)
            module = import_module(model['module'])
            fn = getattr(module, model['name'])
            args = model.get('args', [])
            kwargs = model.get('kwargs', {})
            _logger.debug('Loaded model from {}', path)
            return fn(*args, **kwargs)
        except:
            _logger.error('Could not load the model', path)


class CheckpointLoader(Loader):
    def __call__(self):
        return self.load()

    def _get_last_ckpt_path(self):
        path = p.join(self._save_path, '.ledger.json')
        with io.open(path, 'r') as f:
            ledger = json.load(f)
            return ledger[self._filename]

    def _get_ckpt_path_by(self, criterion):
        path = p.join(
            self._save_path,
            '{}.ckpt-{}'.format(self._filename, criterion))
        return path

    def load(self):
        path = p.join(self._save_path, self._filename)
        try:
            state = self.load_binary(path)
            _logger.debug('Loaded checkpoint from {}', path)
            return state
        except:
            _logger.error('Could not load the checkpoint', path)

    def load_last(self):
        path = self._get_last_ckpt_path()
        try:
            state = self.load_binary(path)
            _logger.debug('Loaded last checkpoint from {}', path)
            return state
        except:
            _logger.error('Could not load the checkpoint', path)

    def load_by(self, criterion):
        path = self._get_ckpt_path_by(criterion)
        try:
            state = self.load_binary(path)
            _logger.debug('Loaded {} checkpoint from {}', criterion, path)
            return state
        except:
            _logger.error('Could not load the checkpoint', path)


class ModelCheckpointLoader(CheckpointLoader):
    def __init__(self, save_path, name='model'):
        super(ModelCheckpointLoader, self).__init__(save_path, name)


class TrainerCheckpointLoader(CheckpointLoader):
    def __init__(self, save_path, name='trainer'):
        super(TrainerCheckpointLoader, self).__init__(save_path, name)

    def load(self):
        state = super(TrainerCheckpointLoader, self).load()
        set_rng_state(state.pop('rng_state'))
        return state

    def load_last(self):
        state = super(TrainerCheckpointLoader, self).load_last()
        set_rng_state(state.pop('rng_state'))
        return state

    def load_by(self, criterion):
        state = super(TrainerCheckpointLoader, self).load_by(criterion)
        set_rng_state(state.pop('rng_state'))
        return state
