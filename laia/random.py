from __future__ import absolute_import

import numpy as np
import torch

import random


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
