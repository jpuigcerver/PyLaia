from __future__ import absolute_import

__all__ = ['data', 'decoders', 'distorter', 'engine', 'losses', 'manual_seed',
           'meters', 'utils']

import numpy as np
import torch

import laia.data
import laia.decoders
import laia.engine
import laia.losses
import laia.meters
import laia.utils

def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
