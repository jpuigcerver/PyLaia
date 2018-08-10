from nnutils_pytorch import adaptive_avgpool_2d

from laia.nn.adaptive_pool_2d_base import AdaptivePool2dBase


class AdaptiveAvgPool2d(AdaptivePool2dBase):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__(
            output_sizes=output_size, func=adaptive_avgpool_2d
        )
