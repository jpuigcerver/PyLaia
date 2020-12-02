from typing import Callable, Sequence, Tuple, Union

import torch

Loss = Union[float, torch.FloatTensor]

ParamNd = Union[int, Sequence[int], torch.LongTensor]
Param2d = Union[int, Tuple[int, int], torch.LongTensor]

Module = Callable[..., torch.nn.Module]
