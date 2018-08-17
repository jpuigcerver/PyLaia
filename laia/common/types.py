from typing import Union, Sequence, Tuple, Callable

import torch

Loss = Union[float, torch.FloatTensor]

ParamNd = Union[int, Sequence[int]]
Param2d = Union[int, Tuple[int, int]]

Module = Callable[..., torch.nn.Module]
