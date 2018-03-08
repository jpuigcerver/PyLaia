import numpy as np
import torch
import unittest

from dortmund_utils import DortmundBCELoss
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss


class DortmundBCELossTest(unittest.TestCase):
    def test(self):
        x = Variable(torch.Tensor(5, 10).normal_(), requires_grad=True)

        y = torch.Tensor(5, 10).normal_()
        y[y >= 0] = 1
        y[y < 0] = 0
        y = Variable(y)

        mref = BCEWithLogitsLoss(size_average=False)
        loss = mref(x, y)
        loss.backward()
        loss_ref = loss.data[0]
        grad_ref = x.grad.data.numpy()

        mdor = DortmundBCELoss()
        loss = mdor(x, y)
        loss_dor = loss.data[0]
        grad_dor = x.grad.data.numpy()

        np.testing.assert_allclose(loss_dor, loss_ref / 5)
        np.testing.assert_allclose(grad_dor, grad_ref / 5)


if __name__ == '__main__':
    unittest.main()