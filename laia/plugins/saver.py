from __future__ import absolute_import

import os

import torch

import laia.plugins.logging as log

BEST_TRAIN_CER = 'lowest-train-cer'
BEST_TRAIN_LOSS = 'lowest-train-loss'
BEST_VALID_CER = 'lowest-valid-cer'
BEST_VALID_LOSS = 'lowest-valid-loss'

_criteria = (BEST_TRAIN_CER, BEST_TRAIN_LOSS,
             BEST_VALID_CER, BEST_VALID_LOSS)


def save_model(model, save_path,
               filename='model', step=None):
    '''
    model: {
        'module': module
        'name': name
        'args': args
        'kwargs': kwargs
    }
    '''
    raise NotImplementedError()


def load_model():
    raise NotImplementedError()


def _get_ckpt_path(save_path, filename, suffix):
    if os.path.split(filename)[0]:
        raise ValueError("'filename' must not be a path")
    ckpt_file = '{}.ckpt-{}'.format(filename, suffix)
    return os.path.join(save_path, ckpt_file)


def _get_last_ckpt(save_path, filename):
    import re
    files = [file for file in os.listdir(save_path)
             if file.startswith(filename + '.ckpt-')
             and all(x not in file for x in _criteria)]
    if len(files) == 0:
        raise LookupError('No model checkpoint found')
    last = None
    highest = -float('inf')
    for file in files:
        match = re.search('.ckpt-(.*)', file)
        suffix = int(match.group(1))
        if suffix > highest:
            highest = suffix
            last = file
    return last


def _get_best_ckpt(save_path, criterion, filename):
    files = [file for file in os.listdir(save_path)
             if file == '{}.ckpt-{}'.format(filename, criterion)]
    if len(files) == 0:
        raise LookupError('No model checkpoint found')
    return files[0]


def save_model_ckpt(state, save_path,
                    filename='model', suffix=0,
                    *args, **kwargs):
    ckpt_path = _get_ckpt_path(save_path, filename, suffix=suffix)
    return torch.save(state, ckpt_path, *args, **kwargs)


def load_model_ckpt(save_path, filename, *args, **kwargs):
    ckpt_path = os.path.join(save_path, filename)
    log.debug('Loaded checkpoint {}', ckpt_path)
    return torch.load(ckpt_path, *args, **kwargs)


def load_last_model_ckpt(save_path, filename='model',
                         *args, **kwargs):
    f = _get_last_ckpt(save_path, filename)
    return load_model_ckpt(save_path, f, *args, **kwargs)


def load_best_model_ckpt(save_path, criterion, filename='model',
                         *args, **kwargs):
    f = _get_best_ckpt(save_path, criterion, filename)
    return load_model_ckpt(save_path, f, *args, **kwargs)


def save_trainer_ckpt(state, save_path,
                      filename='trainer', suffix=0,
                      *args, **kwargs):
    '''
    state: {
        'epoch': epoch,
        'learning_rate': learning_rate,
        'rng': get_rng_state()
        'triggers': triggers
        'args': args
        'kwargs': kwargs
    }
    '''
    raise NotImplementedError()


def load_trainer_ckpt(save_path, filename,
                      *args, **kwargs):
    raise NotImplementedError()


def load_last_trainer_ckpt(save_path, filename='trainer',
                           *args, **kwargs):
    f = _get_last_ckpt(save_path, filename)
    return load_trainer_ckpt(save_path, f, *args, **kwargs)


def load_best_trainer_ckpt(save_path, criterion, filename='trainer',
                           *args, **kwargs):
    f = _get_best_ckpt(save_path, criterion, filename)
    return load_trainer_ckpt(save_path, f, *args, **kwargs)
