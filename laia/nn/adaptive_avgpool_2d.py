from __future__ import absolute_import

from laia.nn.adaptive_pool_2d_base import AdaptivePool2dBase
from nnutils_pytorch import adaptive_avgpool_2d

class AdaptiveAvgPool2d(AdaptivePool2dBase):
    def __init__(self, output_sizes):
        super(AdaptiveAvgPool2d, self).__init__(output_sizes=output_sizes,
                                                func=adaptive_avgpool_2d)
