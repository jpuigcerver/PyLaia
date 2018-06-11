from __future__ import absolute_import

from torch.nn import BCEWithLogitsLoss


class DortmundBCELoss(BCEWithLogitsLoss):
    def __init__(self):
        super(DortmundBCELoss, self).__init__(size_average=False)

    def forward(self, output, target):
        loss = super(DortmundBCELoss, self).forward(output, target)
        return loss / output.size(0)
