from __future__ import absolute_import

import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, targets, **kwargs):
        raise NotImplementedError
