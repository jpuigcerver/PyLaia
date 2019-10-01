import unittest

import torch
from torch.nn import BCEWithLogitsLoss

from laia.losses.dortmund_bce_loss import DortmundBCELoss


class DortmundBCELossTest(unittest.TestCase):
    def test(self):
        x1 = torch.empty(5, 10).normal_().requires_grad_()
        x2 = x1.detach().requires_grad_()

        y = torch.empty(5, 10).normal_()
        y[y >= 0] = 1
        y[y < 0] = 0

        mref = BCEWithLogitsLoss(reduction="sum")
        loss_ref = mref(x1, y)
        loss_ref.backward()

        mdor = DortmundBCELoss()
        loss_dor = mdor(x2, y)
        loss_dor.backward()

        torch.testing.assert_allclose(loss_ref / 5, loss_dor)
        torch.testing.assert_allclose(x1.grad / 5, x2.grad)


if __name__ == "__main__":
    unittest.main()
