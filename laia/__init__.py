from __future__ import absolute_import

__all__ = ['data', 'decoders', 'distorter', 'engine', 'losses', 'manual_seed',
           'meters', 'plugins', 'utils']

import numpy as np
import torch
import random

import laia.data
import laia.decoders
import laia.engine
import laia.losses
import laia.meters
import laia.plugins
import laia.utils


def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_rng_state():
    return {'np': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'python': random.getstate()}


def set_rng_state(state):
    np.random.set_state(state['np'])
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
