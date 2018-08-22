from __future__ import absolute_import

import random
from typing import Optional

import numpy as np
import torch


def manual_seed(seed):
    # type: (int) -> None
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_rng_state(device=None):
    # type: (Optional[torch.Device]) -> dict
    return {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device=device.index)
        if device.type == "cuda"
        else None,
        "python": random.getstate(),
    }


def set_rng_state(state, device=None):
    # type: (dict, Optional[torch.Device]) -> None
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(
            state["torch_cuda"], device=device.index if device else -1
        )
    random.setstate(state["python"])
