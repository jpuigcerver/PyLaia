from __future__ import absolute_import

from laia.nn.adaptive_pool_2d_base import AdaptivePool2dBase
from nnutils_pytorch import adaptive_maxpool_2d

class AdaptiveMaxPool2d(AdaptivePool2dBase):
    def __init__(self, output_sizes):
        super(AdaptiveMaxPool2d, self).__init__(output_sizes=output_sizes,
                                                func=adaptive_maxpool_2d)
