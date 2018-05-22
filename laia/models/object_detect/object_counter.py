from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


class ObjectCounter(nn.Module):

    def __init__(
        self,
        num_input_channels,
        pooling_size,
        lin_units,
        lin_batch_norm,
        lin_dropout,
        lin_activation,
        pooling="max",
        inplace=False,
    ):
        super(ObjectCounter, self).__init__()
        pooling_funcs = {"avg": F.adaptive_avgpool_2d, "max": F.adaptive_maxpool_2d}
        self._pooling_size = pooling_size
        self._pooling = pooling_funcs.get(pooling.lower(), None)
        if self._pooling is None:
            raise NotImplementedError('Unknown pooling method: "%s"' % pooling)

        ni = num_input_channels
        for i, (nh, bn, dr, af) in enumerate(
            zip(lin_units, lin_batch_norm, lin_dropout, lin_activation)
        ):
            if dr and dr > 0.0:
                self.add_module("dropout%d" % i, nn.Dropout(dr, inplace=inplace))
            self.add_module("linear%d" % i, nn.Linear(ni, nh))
            if bn:
                self.add_module("batchnorm%d" % i, nn.BatchNorm2d(nh))
            if inplace:
                # Activation function must support inplace operations.
                self.add_module("activation%d" % i, af(inplace=True))
            else:
                self.add_module("activation%d" % i, af())
            ni = nh

        self.add_module("log_mean_var", nn.Linear(ni, 2))

    def forward(self, x):
        x = self._pooling(x, self._pooling_size)
        for module in self._modules.values():
            x = module(x)
        return x
