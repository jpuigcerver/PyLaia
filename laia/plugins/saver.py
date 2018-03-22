from __future__ import absolute_import

import io
import json
import os
from importlib import import_module

import torch

from laia.random import get_rng_state, set_rng_state
from laia.logging import get_logger

_logger = get_logger(__name__)

BEST_TRAIN_CER = 'lowest-train-cer'
BEST_TRAIN_LOSS = 'lowest-train-loss'
BEST_VALID_CER = 'lowest-valid-cer'
BEST_VALID_LOSS = 'lowest-valid-loss'

_criteria = (BEST_TRAIN_CER, BEST_TRAIN_LOSS,
             BEST_VALID_CER, BEST_VALID_LOSS)


def save_model(model, save_path, filename='model',
               *args, **kwargs):
    """Saves the model represented by a dict as a json

    Args:
        model (dict): Dictionary representing the model. It's structure
        is defined as::
            {
                'module': module string containing the class/function
                    which created the model
                'name': class/function which created the model
                'args' (optional): class/function args
                'kwargs' (optional): class/function kwargs
            }
        save_path (str): Path in which the file will be saved
        filename (str, optional): Name of the file. Defaults to 'model'
        *args: `json.dump` args
        **kwargs: `json.dump` kwargs
    """
    full_path = os.path.join(save_path, filename)
    with io.open(full_path, 'w') as f:
        json.dump(model, f, *args, **kwargs)
    _logger.debug('Saved model to {}', full_path)


def load_model(save_path, filename='model'):
    """Loads a model from a json file

    The model structure is described in ``save_model``

    Args:
        save_path (str): Path in which the model file is located
        filename (str, optional): Name of the file. Defaults to 'model'

    Returns:
        The class/function called by ``model['name']`` located in
        ``model['module']`` with its args and kwargs applied

    """
    full_path = os.path.join(save_path, filename)
    with io.open(full_path, 'r') as f:
        model = json.load(f)
    module = import_module(model['module'])
    fn = getattr(module, model['name'])
    args, kwargs = model.get('args'), model.get('kwargs')
    _logger.debug('Loaded model from {}', full_path)
    return fn(*args, **kwargs)


def _get_ckpt_path(save_path, filename, suffix):
    """Returns the checkpoint path"""
    if os.path.split(filename)[0]:
        raise ValueError("'filename' must not be a path")
    ckpt_file = filename + '.ckpt'
    if suffix is not None:
        ckpt_file = '{}-{}'.format(ckpt_file, suffix)
    return os.path.join(save_path, ckpt_file)


def _get_last_ckpt(save_path, filename):
    """Returns the filename of the last checkpoint.
    The last checkpoint is searched looking at the highest suffix,
    (e.g. model.ckpt-1000 > model.ckpt-10) or no suffix at all.
    """
    if os.path.split(filename)[0]:
        raise ValueError("'filename' must not be a path")
    files = os.listdir(save_path)
    if len(files) == 0:
        raise LookupError('No files found in the directory')
    pattern = filename + '.ckpt-(\d)+'
    import re
    suffixes = [int(re.search(pattern, f).group(1)) for f in files]
    if len(suffixes) == 0:
        raise LookupError('No checkpoint found with numeric suffix')
    last = max(suffixes)
    for file in files:
        if file.endswith('.ckpt-{}'.format(last)):
            return file
    if filename + '.ckpt' in files:
        return filename + '.ckpt'
    return None


def _get_best_ckpt(save_path, criterion, filename):
    """Returns the checkpoint filename associated
     to the given criterion"""
    files = [file for file in os.listdir(save_path)
             if file == '{}.ckpt-{}'.format(filename, criterion)]
    if len(files) == 0:
        raise LookupError('No model checkpoint found')
    return files[0]


def save_model_ckpt(state, save_path,
                    filename='model', suffix=None,
                    *args, **kwargs):
    ckpt_path = _get_ckpt_path(save_path, filename, suffix=suffix)
    s = torch.save(state, ckpt_path, *args, **kwargs)
    _logger.debug('Saved model checkpoint to {}', ckpt_path)
    return s


def load_model_ckpt(save_path, filename, *args, **kwargs):
    ckpt_path = os.path.join(save_path, filename)
    l = torch.load(ckpt_path, *args, **kwargs)
    _logger.debug('Loaded model checkpoint from {}', ckpt_path)
    return l


def load_last_model_ckpt(save_path, filename='model',
                         *args, **kwargs):
    f = _get_last_ckpt(save_path, filename)
    return load_model_ckpt(save_path, f, *args, **kwargs)


def load_best_model_ckpt(save_path, criterion, filename='model',
                         *args, **kwargs):
    f = _get_best_ckpt(save_path, criterion, filename)
    return load_model_ckpt(save_path, f, *args, **kwargs)


def save_trainer_ckpt(state, save_path,
                      filename='trainer', suffix=None,
                      *args, **kwargs):
    '''
    state: {
        'epoch': epoch,
        'optim_state_dict': optim_state_dict,
        'rng': get_rng_state()
        'triggers': triggers
        'args': args
        'kwargs': kwargs
    }
    '''
    ckpt_path = _get_ckpt_path(save_path, filename, suffix=suffix)
    state['rng'] = get_rng_state()
    s = torch.save(state, ckpt_path, *args, **kwargs)
    _logger.debug('Saved trainer checkpoint to {}', ckpt_path)
    return s


def load_trainer_ckpt(save_path, filename,
                      *args, **kwargs):
    ckpt_path = os.path.join(save_path, filename)
    l = torch.load(ckpt_path, *args, **kwargs)
    set_rng_state(l.get['rng'])
    _logger.debug('Loaded model checkpoint from {}', ckpt_path)
    return l


def load_last_trainer_ckpt(save_path, filename='trainer',
                           *args, **kwargs):
    f = _get_last_ckpt(save_path, filename)
    return load_trainer_ckpt(save_path, f, *args, **kwargs)


def load_best_trainer_ckpt(save_path, criterion, filename='trainer',
                           *args, **kwargs):
    f = _get_best_ckpt(save_path, criterion, filename)
    return load_trainer_ckpt(save_path, f, *args, **kwargs)
