import unittest

import torch

from laia.data import PaddedTensor
from laia.models.kws.dortmund_phocnet import (
    build_conv_model,
    size_after_conv,
    DortmundPHOCNet,
)


class DortmundPHOCNetTest(unittest.TestCase):
    def test_size_after_conv(self):
        m = build_conv_model()
        for x in [1, 1], [3, 3], [4, 4], [9, 9], [37, 23]:
            y = m(torch.empty(1, 1, *x))
            ys = list(y.size()[-2:])
            expected_ys = size_after_conv(torch.tensor([x])).tolist()[0]
            self.assertEqual(expected_ys, ys)

    def test_number_parameters(self):
        m = DortmundPHOCNet(phoc_size=540, tpp_levels=[1, 2, 3, 4, 5])
        self.assertEqual(59859420, sum(p.numel() for p in m.parameters()))

    def test_single_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, tpp_levels=[1, 2, 3])
        x = torch.randn(1, 1, 93, 30)
        y = m(x)
        self.assertEqual([1, 125], list(y.size()))

    def test_batch_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, tpp_levels=[1, 2, 3])
        x = torch.randn(3, 1, 93, 30)
        y = m(x)
        self.assertEqual([3, 125], list(y.size()))

    def test_padded_batch_output_size(self):
        m = DortmundPHOCNet(phoc_size=125, tpp_levels=[1, 2, 3])
        x = torch.randn(3, 1, 93, 30)
        xs = torch.tensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(x, xs))
        self.assertEqual([3, 125], list(y.size()))

    def test_padded_batch_output_size_spp(self):
        m = DortmundPHOCNet(phoc_size=10, tpp_levels=None, spp_levels=[1, 2])
        x = torch.randn(3, 1, 93, 30)
        xs = torch.tensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(x, xs))
        self.assertEqual([3, 10], list(y.size()))

    def test_padded_batch_output_size_tpp_and_spp(self):
        m = DortmundPHOCNet(phoc_size=40, tpp_levels=[1, 2, 3], spp_levels=[1, 2])
        x = torch.randn(3, 1, 93, 30)
        xs = torch.tensor([[93, 30], [40, 30], [93, 20]])
        y = m(PaddedTensor(x, xs))
        self.assertEqual([3, 40], list(y.size()))

    def test_single_grad(self):
        # TODO(jpuigcerver): Find a way to properly check the model end-to-end
        pass
