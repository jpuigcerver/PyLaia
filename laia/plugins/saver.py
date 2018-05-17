from __future__ import absolute_import

import inspect
import os
from collections import deque

import torch

import laia
from laia.logging import get_logger
from laia.random import get_rng_state

_logger = get_logger(__name__)


class Saver(object):
    def __call__(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    def save(self, *args, **kwargs):
        raise NotImplementedError


class BasicSaver(Saver):
    def save(self, obj, filepath):
        dirname = os.path.dirname(os.path.normpath(filepath))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(obj, filepath)
        return filepath


class ObjectSaver(Saver):
    def __init__(self, filepath):
        self._filepath = filepath
        self._basic_saver = BasicSaver()

    def save(self, func_or_class, *args, **kwargs):
        return self._basic_saver.save({
            'module': inspect.getmodule(func_or_class).__name__,
            'name': func_or_class.__name__,
            'args': args,
            'kwargs': kwargs}, self._filepath)


class ModelSaver(ObjectSaver):
    def __init__(self, save_path, filename='model'):
        super(ModelSaver, self).__init__(os.path.join(save_path, filename))

    def save(self, func, *args, **kwargs):
        path = super(ModelSaver, self).save(func, *args, **kwargs)
        _logger.debug('Saved model {}', path)
        return path


class TrainerSaver(ObjectSaver):
    def __init__(self, save_path, filename='trainer'):
        super(TrainerSaver, self).__init__(os.path.join(save_path, filename))

    def save(self, func, *args, **kwargs):
        path = super(TrainerSaver, self).save(func, *args, **kwargs)
        _logger.debug('Saved trainer {}', path)
        return path


class CheckpointSaver(Saver):
    def __init__(self, filepath):
        self._filepath = filepath
        self._basic_saver = BasicSaver()

    def get_ckpt(self, suffix):
        ckpt_filepath = '{}-{}'.format(self._filepath, suffix) \
            if suffix is not None else self._filepath
        return ckpt_filepath

    def save(self, state, suffix=None):
        path = self._basic_saver.save(state, self.get_ckpt(suffix))
        _logger.debug('Saved checkpoint {}', path)
        return path


class ModelCheckpointSaver(Saver):
    def __init__(self, ckpt_saver, model):
        # type: (CheckpointSaver, torch.nn.Module) -> None
        self._ckpt_saver = ckpt_saver
        self._model = model

    def save(self, suffix=None):
        return self._ckpt_saver.save(self._model.state_dict(), suffix=suffix)


class TrainerCheckpointSaver(Saver):
    def __init__(self, ckpt_saver, trainer, gpu=None):
        self._ckpt_saver = ckpt_saver
        self._trainer = trainer
        self._gpu = gpu

    def save(self, suffix=None):
        state = dict(rng_state=get_rng_state(gpu=self._gpu),
                     **self._trainer.state_dict())
        return self._ckpt_saver.save(state, suffix=suffix)


class RollingSaver(Saver):
    """Saver wrapper that keeps a maximum number of files"""

    def __init__(self, saver, keep=5):
        # type: (Saver, int) -> None
        assert keep > 0
        self._saver = saver
        self._keep = keep
        self._last_saved = deque()

    def save(self, *args, **kwargs):
        path = self._saver.save(*args, **kwargs)
        if len(self._last_saved) >= self._keep:
            last = self._last_saved.popleft()
            try:
                os.remove(last)
                _logger.debug('{} checkpoint removed', last)
            except OSError:
                # Someone else removed the checkpoint, not a big deal
                pass
        self._last_saved.append(path)
        return path
