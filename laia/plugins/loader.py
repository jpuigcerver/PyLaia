from __future__ import absolute_import

import io
import json
import os.path as p
from importlib import import_module

import torch

from laia.plugins.logging import get_logger

_logger = get_logger(__name__)


class Loader(object):
    def __init__(self, save_path, filename):
        assert not p.dirname(filename)
        assert p.isfile(p.join(save_path, filename))
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
    def load(self):
        path = p.join(self._save_path, self._filename)
        try:
            model = Loader.load_json(path)
            module = import_module(model['module'])
            fn = getattr(module, model['name'])
            args, kwargs = model.get('args'), model.get('kwargs')
            _logger.debug('Loaded model from {}', path)
            return fn(*args, **kwargs)
        except:
            _logger.error('Could not load the model', path)


class CheckpointLoader(Loader):
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
            state = Loader.load_binary(path)
            _logger.debug('Loaded checkpoint from {}', path)
            return state
        except:
            _logger.error('Could not load the checkpoint', path)

    def load_last(self):
        path = self._get_last_ckpt_path()
        try:
            state = Loader.load_binary(path)
            _logger.debug('Loaded last checkpoint from {}', path)
            return state
        except:
            _logger.error('Could not load the checkpoint', path)

    def load_by(self, criterion):
        path = self._get_ckpt_path_by(criterion)
        try:
            state = Loader.load_binary(path)
            _logger.debug('Loaded {} checkpoint from {}', criterion, path)
            return state
        except:
            _logger.error('Could not load the checkpoint', path)


class ModelCheckpointLoader(CheckpointLoader):
    def __init__(self, save_path, filename='model'):
        super(ModelCheckpointLoader, self).__init__(save_path, filename)

    # TODO: Override load here?


class TrainerCheckpointLoader(CheckpointLoader):
    def __init__(self, save_path, filename='trainer'):
        super(TrainerCheckpointLoader, self).__init__(save_path, filename)

    def load(self):
        # TODO
        raise NotImplemented

    # TODO: load_last and load_by?
