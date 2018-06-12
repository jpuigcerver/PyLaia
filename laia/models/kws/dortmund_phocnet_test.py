from __future__ import absolute_import

import unittest

import torch
from laia.data import PaddedTensor
from laia.models.kws.dortmund_phocnet import DortmundPHOCNet
from torch.autograd import Variable


class DortmundPHOCNetTest(unittest.TestCase):
    def test_number_parameters(self):
        m = DortmundPHOCNet(phoc_size=540, pyramid_levels=5)
        num_parameters = sum([p.numel() for p in m.parameters()])
        self.assertEqual(num_parameters, 59859420)

    def test_single_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, pyramid_levels=3)
        x = torch.randn(1, 1, 93, 30)
        y = m(Variable(x))
        ys = list(y.size())
        self.assertListEqual(ys, [1, 125])

    def test_batch_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, pyramid_levels=3)
        x = torch.randn(3, 1, 93, 30)
        y = m(Variable(x))
        ys = list(y.size())
        self.assertListEqual(ys, [3, 125])

    def test_padded_batch_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, pyramid_levels=3)
        x = torch.randn(3, 1, 93, 30)
        xs = torch.LongTensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(Variable(x), Variable(xs)))
        ys = list(y.size())
        self.assertListEqual(ys, [3, 125])

    def test_single_grad(self):
        # TODO: Find a way of properly checking the model end-to-end
        pass
        """
        m = DortmundPHOCNet(phoc_size=125, pyramid_levels=1, test=True)
        m.train()

        x = Variable(torch.randn(1, 1, 5, 3), requires_grad=True)

        def warp_func(xx):
            return m(xx).sum()

        gradcheck(func=warp_func, inputs=(x,))
        """
