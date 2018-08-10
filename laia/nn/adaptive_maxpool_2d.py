from nnutils_pytorch import adaptive_maxpool_2d

from laia.nn.adaptive_pool_2d_base import AdaptivePool2dBase


class AdaptiveMaxPool2d(AdaptivePool2dBase):
    def __init__(self, output_size):
        super(AdaptiveMaxPool2d, self).__init__(
            output_sizes=output_size, func=adaptive_maxpool_2d
        )
