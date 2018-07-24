from __future__ import absolute_import

import unittest

import torch
from torch.autograd import Variable

from laia.data import PaddedTensor
from laia.models.kws.dortmund_phocnet import (
    build_conv_model,
    size_after_conv,
    DortmundPHOCNet,
)


class DortmundPHOCNetTest(unittest.TestCase):
    def test_size_after_conv(self):
        m = build_conv_model()
        y = m(Variable(torch.Tensor(1, 1, 1, 1)))
        actual_ys = list(y.size()[-2:])
        expected_ys = size_after_conv(torch.LongTensor([[1, 1]])).tolist()[0]
        self.assertListEqual(actual_ys, expected_ys)

        y = m(Variable(torch.Tensor(1, 1, 3, 3)))
        actual_ys = list(y.size()[-2:])
        expected_ys = size_after_conv(torch.LongTensor([[3, 3]])).tolist()[0]
        self.assertListEqual(actual_ys, expected_ys)

        y = m(Variable(torch.Tensor(1, 1, 4, 4)))
        actual_ys = list(y.size()[-2:])
        expected_ys = size_after_conv(torch.LongTensor([[4, 4]])).tolist()[0]
        self.assertListEqual(actual_ys, expected_ys)

        y = m(Variable(torch.Tensor(1, 1, 9, 9)))
        actual_ys = list(y.size()[-2:])
        expected_ys = size_after_conv(torch.LongTensor([[9, 9]])).tolist()[0]
        self.assertListEqual(actual_ys, expected_ys)

        y = m(Variable(torch.Tensor(1, 1, 37, 23)))
        actual_ys = list(y.size()[-2:])
        expected_ys = size_after_conv(torch.LongTensor([[37, 23]])).tolist()[0]
        self.assertListEqual(actual_ys, expected_ys)

    def test_number_parameters(self):
        m = DortmundPHOCNet(phoc_size=540, tpp_levels=[1, 2, 3, 4, 5])
        num_parameters = sum([p.numel() for p in m.parameters()])
        self.assertEqual(num_parameters, 59859420)

    def test_single_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, tpp_levels=[1, 2, 3])
        x = torch.randn(1, 1, 93, 30)
        y = m(Variable(x))
        ys = list(y.size())
        self.assertListEqual(ys, [1, 125])

    def test_batch_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, tpp_levels=[1, 2, 3])
        x = torch.randn(3, 1, 93, 30)
        y = m(Variable(x))
        ys = list(y.size())
        self.assertListEqual(ys, [3, 125])

    def test_padded_batch_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, tpp_levels=[1, 2, 3])
        x = torch.randn(3, 1, 93, 30)
        xs = torch.LongTensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(Variable(x), Variable(xs)))
        ys = list(y.size())
        self.assertListEqual(ys, [3, 125])

    def test_padded_batch_output_size_spp(self):
        m = DortmundPHOCNet(phoc_size=10, tpp_levels=None, spp_levels=[1, 2])
        x = torch.randn(3, 1, 93, 30)
        xs = torch.LongTensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(Variable(x), Variable(xs)))
        ys = list(y.size())
        self.assertListEqual(ys, [3, 10])

    def test_padded_batch_output_size_tpp_and_spp(self):
        m = DortmundPHOCNet(phoc_size=40, tpp_levels=[1, 2, 3], spp_levels=[1, 2])
        x = torch.randn(3, 1, 93, 30)
        xs = torch.LongTensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(Variable(x), Variable(xs)))
        ys = list(y.size())
        self.assertListEqual(ys, [3, 40])

    def test_single_grad(self):
        # TODO: Find a way to properly check the model end-to-end
        pass
        """
        m = DortmundPHOCNet(phoc_size=125, pyramid_levels=1, test=True)
        m.train()

        x = Variable(torch.randn(1, 1, 5, 3), requires_grad=True)

        def warp_func(xx):
            return m(xx).sum()

        gradcheck(func=warp_func, inputs=(x,))
        """
