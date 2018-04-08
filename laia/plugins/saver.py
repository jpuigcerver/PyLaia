from __future__ import absolute_import

import inspect
import io
import json
import os
from collections import deque

import os.path as p
import torch

from laia.logging import get_logger
from laia.random import get_rng_state

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

_logger = get_logger(__name__)


class Saver(object):
    def __init__(self, save_path, filename):
        assert not p.dirname(filename)
        assert p.exists(save_path)
        self._save_path = p.normpath(save_path)
        self._filename = filename
        self._path = p.join(self._save_path, self._filename)

    @property
    def path(self):
        return self._path

    def save_binary(self, obj, path):
        try:
            torch.save(obj, path)
            self._update_ledger(path)
        except FileNotFoundError:
            _logger.info('Could not find the file {}', self.path)

    def _update_ledger(self, path):
        ledger_path = p.join(self._save_path, '.ledger.json')
        if os.path.isfile(ledger_path):
            with io.open(ledger_path) as f:
                ledger = json.load(f)
        else:
            ledger = {}
        ledger[self._filename] = path
        with io.open(ledger_path, 'w') as f:
            json.dump(ledger, f)


class ObjectSaver(Saver):
    def __call__(self, func, *args, **kwargs):
        return self.save(func, *args, **kwargs)

    def save(self, func, *args, **kwargs):
        self.save_binary({
            'module': inspect.getmodule(func).__name__,
            'name': func.__name__,
            'args': args,
            'kwargs': kwargs
        }, self.path)


class ModelSaver(ObjectSaver):
    def __init__(self, save_path, filename='model'):
        super(ModelSaver, self).__init__(save_path, filename)

    def save(self, func, *args, **kwargs):
        try:
            super(ModelSaver, self).save(func, *args, **kwargs)
            _logger.debug('Saved model {}', self.path)
        except Exception as e:
            _logger.error('Error while saving the model {}: {}', self.path, e)


class TrainerSaver(ObjectSaver):
    def __init__(self, save_path, filename='trainer'):
        super(TrainerSaver, self).__init__(save_path, filename)

    def save(self, func, *args, **kwargs):
        try:
            super(TrainerSaver, self).save(func, *args, **kwargs)
            _logger.debug('Saved trainer {}', self.path)
        except Exception as e:
            _logger.error('Error while saving the trainer {}: {}', self.path, e)


class CheckpointSaver(Saver):
    def __call__(self, state, suffix=None):
        return self.save(state, suffix=suffix)

    def _get_ckpt_path(self, suffix=None):
        ckpt_file = '{}-{}'.format(self._filename, suffix) if suffix else self._filename
        return p.join(self._save_path, ckpt_file)

    def save(self, state, suffix=None):
        path = self._get_ckpt_path(suffix=suffix)
        try:
            self.save_binary(state, path)
            _logger.debug('Saved checkpoint {}', path)
        except Exception as e:
            _logger.error('Error while saving the checkpoint {}: {}', path, e)


class ModelCheckpointSaver(CheckpointSaver):
    def __init__(self, save_path, filename='model.ckpt'):
        super(ModelCheckpointSaver, self).__init__(save_path, filename)


class TrainerCheckpointSaver(CheckpointSaver):
    def __init__(self, save_path, filename='trainer.ckpt'):
        super(TrainerCheckpointSaver, self).__init__(save_path, filename)

    def save(self, state, suffix=None):
        state['rng_state'] = get_rng_state()
        super(TrainerCheckpointSaver, self).save(state, suffix=suffix)


class LastCheckpointsSaver(object):
    def __init__(self, checkpoint_saver, keep_checkpoints=5):
        assert keep_checkpoints > 0
        self._ckpt_saver = checkpoint_saver
        self._keep_ckpts = keep_checkpoints
        self._last_ckpts = deque()
        self._ckpt_num = 0

    def __call__(self, state, suffix=None):
        return self.save(state, suffix=suffix)

    def save(self, state, suffix=None):
        path = self._ckpt_saver.save(state, suffix=suffix)
        if len(self._last_ckpts) < self._keep_ckpts:
            self._last_ckpts.append(path)
        else:
            last = self._last_ckpts.popleft()
            try:
                os.remove(last)
                _logger.debug('{} checkpoint removed', last)
            except Exception as e:
                _logger.error('Error while removing the checkpoint {}: {}', last, e)
            self._last_ckpts.append(path)
            self._ckpt_num = (self._ckpt_num + 1) % self._keep_ckpts
