from __future__ import absolute_import

import numpy as np
import torch
import unittest

from laia.losses.dortmund_bce_loss import DortmundBCELoss
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss


class DortmundBCELossTest(unittest.TestCase):
    def test(self):
        x1 = Variable(torch.Tensor(5, 10).normal_(), requires_grad=True)
        x2 = Variable(x1.data.clone(), requires_grad=True)

        y = torch.Tensor(5, 10).normal_()
        y[y >= 0] = 1
        y[y < 0] = 0
        y = Variable(y)

        mref = BCEWithLogitsLoss(size_average=False)
        loss = mref(x1, y)
        loss.backward()
        loss_ref = loss.data[0]
        grad_ref = x1.grad.data.cpu().numpy()

        mdor = DortmundBCELoss()
        loss = mdor(x2, y)
        loss.backward()
        loss_dor = loss.data[0]
        grad_dor = x2.grad.data.cpu().numpy()

        np.testing.assert_almost_equal(loss_dor, loss_ref / 5, decimal=6)
        np.testing.assert_almost_equal(grad_dor, grad_ref / 5, decimal=6)


if __name__ == "__main__":
    unittest.main()
