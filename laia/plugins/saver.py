from __future__ import absolute_import

import inspect
import io
import json
import os
from builtins import str
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
            return path
        except FileNotFoundError:
            _logger.info('Could not find the file {}', self.path)
            return None

    def _update_ledger(self, path):
        ledger_path = p.join(self._save_path, '.ledger.json')
        if os.path.isfile(ledger_path):
            with io.open(ledger_path) as f:
                ledger = json.load(f)
        else:
            ledger = {}
        ledger[self._filename] = path
        with io.open(ledger_path, 'w') as f:
            f.write(str(json.dumps(ledger)))


class ObjectSaver(Saver):
    def __call__(self, func, *args, **kwargs):
        return self.save(func, *args, **kwargs)

    def save(self, func, *args, **kwargs):
        return self.save_binary({
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
            path = super(ModelSaver, self).save(func, *args, **kwargs)
            _logger.debug('Saved model {}', path)
            return path
        except Exception as e:
            _logger.error('Error while saving the model {}: {}', self.path, e)
            return None


class TrainerSaver(ObjectSaver):
    def __init__(self, save_path, filename='trainer'):
        super(TrainerSaver, self).__init__(save_path, filename)

    def save(self, func, *args, **kwargs):
        try:
            path = super(TrainerSaver, self).save(func, *args, **kwargs)
            _logger.debug('Saved trainer {}', path)
            return path
        except Exception as e:
            _logger.error('Error while saving the trainer {}: {}', self.path, e)
            return None


class CheckpointSaver(Saver):
    def __call__(self, state, suffix=None):
        return self.save(state, suffix=suffix)

    def _get_ckpt_path(self, suffix=None):
        ckpt_file = '{}-{}'.format(self._filename,
                                   suffix) if suffix else self._filename
        return p.join(self._save_path, ckpt_file)

    def save(self, state, suffix=None):
        path = self._get_ckpt_path(suffix=suffix)
        try:
            path = self.save_binary(state, path)
            _logger.debug('Saved checkpoint {}', path)
            return path
        except Exception as e:
            _logger.error('Error while saving the checkpoint {}: {}', path, e)
            return None


class ModelCheckpointSaver(Saver):
    def __init__(self, model, ckpt_saver):
        super(ModelCheckpointSaver, self).__init__(ckpt_saver._save_path,
                                                   ckpt_saver._filename)
        self._model = model
        self._ckpt_saver = ckpt_saver

    def __call__(self, suffix=None):
        return self._ckpt_saver.save(suffix=suffix)

    def save(self, suffix=None):
        return self._ckpt_saver.save(self._model.state_dict(), suffix=suffix)


class TrainerCheckpointSaver(Saver):
    def __init__(self, trainer, ckpt_saver):
        super(TrainerCheckpointSaver, self).__init__(ckpt_saver._save_path,
                                                     ckpt_saver._filename)
        self._trainer = trainer
        self._ckpt_saver = ckpt_saver

    def __call__(self, suffix=None):
        return self._ckpt_saver.save(suffix=suffix)

    def save(self, suffix=None):
        state = dict(rng_state=get_rng_state(), **self._trainer.state_dict())
        return self._ckpt_saver.save(state, suffix=suffix)


class BackupSaver(object):
    def __init__(self, saver, keep=5):
        assert keep > 0
        self._saver = saver
        self._keep = keep
        self._last_saved = deque()

    def __call__(self, suffix=None):
        return self.save(suffix=suffix)

    def save(self, suffix=None):
        path = self._saver.save(suffix=suffix)
        if len(self._last_saved) >= self._keep:
            last = self._last_saved.popleft()
            try:
                os.remove(last)
                _logger.debug('{} checkpoint removed', last)
            except Exception as e:
                _logger.error('Error while removing the checkpoint {}: {}',
                              last, e)
        self._last_saved.append(path)
        return path
