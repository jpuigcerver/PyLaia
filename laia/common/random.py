import random
from typing import Dict, Optional

import numpy as np
import torch


def manual_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_rng_state(device: Optional[torch.device] = None) -> Dict:
    torch_cuda_rng_state = (
        torch.cuda.get_rng_state(device=device.index) if device.type == "cuda" else None
    )
    return {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch_cuda_rng_state,
        "python": random.getstate(),
    }


def set_rng_state(state: Dict, device: Optional[torch.device] = None) -> None:
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(
            state["torch_cuda"].cpu(), device=device.index if device else -1
        )
    random.setstate(state["python"])
