from __future__ import absolute_import

import io
import json
import os.path as p

import torch

from laia.random import get_rng_state, set_rng_state
from laia.logging import get_logger

_logger = get_logger(__name__)


class Saver(object):
    def __init__(self, save_path, filename):
        assert not p.dirname(filename)
        self._save_path = save_path
        self._filename = filename

    @staticmethod
    def save_json(obj, path):
        with io.open(path, 'w') as f:
            json.dump(obj, f)

    def save_binary(self, obj, path):
        torch.save(obj, path)
        self._update_ledger(path)

    def _update_ledger(self, path):
        hidden = p.join(self._save_path, '.ledger.json')
        with io.open(hidden, 'rw') as f:
            ledger = json.load(f) or {}
            ledger[self._filename] = path
            json.dump(ledger, f)


class ModelSaver(Saver):
    def __init__(self, save_path, filename='model'):
        super(ModelSaver, self).__init__(save_path, filename)

    def save(self, model):
        path = p.join(self._save_path, self._filename)
        try:
            Saver.save_json({
                # TODO: Where are these?
                # 'module': model.module,
                # 'name': model.name,
                # 'args': model.args,
                # 'kwargs': model.kwargs
            }, path)
            _logger.debug('Model saved: {}', path)
        except:
            _logger.error('Could not save the model {}', path)
        return path


class CheckpointSaver(Saver):
    def __init__(self, save_path, filename, suffix=None):
        super(CheckpointSaver, self).__init__(save_path, filename)
        self._save_path = save_path
        self._filename = filename
        self._suffix = suffix

    def _get_ckpt_path(self):
        ckpt_file = self._filename + '.ckpt'
        if self._suffix is not None:
            ckpt_file = '{}-{}'.format(ckpt_file, self._suffix)
        return p.join(self._save_path, ckpt_file)

    def save(self, obj):
        path = self._get_ckpt_path()
        try:
            super(CheckpointSaver, self).save_binary(obj, path)
            _logger.debug('Checkpoint saved: {}', path)
        except:
            _logger.error('Could not save the checkpoint {}', path)
        return path


class ModelCheckpointSaver(CheckpointSaver):
    def __init__(self, save_path, filename='model', suffix=None):
        super(ModelCheckpointSaver, self).__init__(save_path, filename, suffix)

    def save(self, model):
        super(ModelCheckpointSaver, self).save(model.state_dict())


class TrainerCheckpointSaver(CheckpointSaver):
    def __init__(self, save_path, filename='trainer', suffix=None):
        super(TrainerCheckpointSaver, self).__init__(save_path, filename, suffix)

    def save(self, trainer):
        super(TrainerCheckpointSaver, self).save({
            'epochs': trainer.epochs,
            'optimizer_state': trainer.optimizer.state_dict(),
            'rng_state': get_rng_state(),
            # TODO: Where are these?
            # 'triggers': trainer.triggers,
            # 'args': trainer.args,
            # 'kwargs': trainer.kwargs
        })


''' TODO
class LastCheckpointsSaver(CheckpointSaver):
    def __init__(self, save_path, filename, suffix=None, keep_checkpoints=5):
        super(LastCheckpointsSaver, self).__init__(save_path, filename, suffix)
        self._keep_ckpts = keep_checkpoints
        self._last_ckpts = []
        self._ckpt_num = 0

    def save(self, obj):
        ckpt_path = super(LastCheckpointsSaver, self).save(obj)
        if len(self._last_ckpts) < self._keep_ckpts:
            self._last_ckpts.append(ckpt_path)
        else:
            last = self._last_ckpts.pop()
            try:
                os.remove(last)
                _logger.debug('{} parameters removed', last)
            except Exception:
                _logger.error('{} parameters could not be removed', last)
            self._last_ckpts.append(ckpt_path)
            self._ckpt_num = (self._ckpt_num + 1) % self._keep_ckpts
        return ckpt_path
'''
