from typing import Union, Sequence, Tuple, Callable

import torch

Loss = Union[float, torch.FloatTensor]

ParamNd = Union[int, Sequence[int], torch.LongTensor]
Param2d = Union[int, Tuple[int, int], torch.LongTensor]

Module = Callable[..., torch.nn.Module]
