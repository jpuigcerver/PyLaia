from __future__ import absolute_import

import random

import numpy as np
import torch


def manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_rng_state(gpu=None):
    return {
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device=gpu - 1) if gpu else None,
        "python": random.getstate(),
    }


def set_rng_state(state, gpu=None):
    np.random.set_state(state["np"])
    if not gpu:
        torch.set_rng_state(state["torch_cpu"])
    elif state["torch_cuda"] is not None:
        # TODO(carmocca): Why is .cpu() necessary?
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=gpu - 1)
    random.setstate(state["python"])
