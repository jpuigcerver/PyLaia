from __future__ import absolute_import

import numpy as np
import torch
import unittest

from laia.losses.dortmund_bce_loss import DortmundBCELoss
from torch.nn import BCEWithLogitsLoss


class DortmundBCELossTest(unittest.TestCase):
    def test(self):
        x1 = torch.empty(5, 10).normal_().requires_grad_()
        x2 = x1.detach().requires_grad_()

        y = torch.empty(5, 10).normal_()
        y[y >= 0] = 1
        y[y < 0] = 0

        mref = BCEWithLogitsLoss(size_average=False)
        loss_ref = mref(x1, y)
        loss_ref.backward()

        mdor = DortmundBCELoss()
        loss_dor = mdor(x2, y)
        loss_dor.backward()

        self.assertTrue(torch.allclose(loss_dor, loss_ref / 5))
        self.assertTrue(torch.allclose(x2.grad, x1.grad / 5))


if __name__ == "__main__":
    unittest.main()
