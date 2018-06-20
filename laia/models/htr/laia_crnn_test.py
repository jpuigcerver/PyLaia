from __future__ import absolute_import
from __future__ import division

import unittest

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

from laia.data import PaddedTensor
from laia.models.htr.laia_crnn import LaiaCRNN


class LaiaCRNNTest(unittest.TestCase):
    def test_get_conv_output_size(self):
        ys = LaiaCRNN.get_conv_output_size(
            size=(30, 40),
            cnn_kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
            cnn_stride=[(1, 1), (1, 1), (1, 1), (1, 1)],
            cnn_dilation=[(1, 1), (1, 1), (1, 1), (1, 1)],
            cnn_poolsize=[(2, 2), (2, 2), (2, 2), (1, 1)],
        )
        self.assertTupleEqual(ys, (30 // 8, 40 // 8))

    def test_fixed_height(self):
        m = LaiaCRNN(
            3,
            30,
            cnn_num_features=[16, 32, 48, 64],
            cnn_kernel_size=[3, 3, 3, 3],
            cnn_stride=[1, 1, 1, 1],
            cnn_dilation=[1, 1, 1, 1],
            cnn_activation=[torch.nn.ReLU] * 4,
            cnn_poolsize=[2, 2, 2, 0],
            cnn_dropout=[0, 0, 0.2, 0.1],
            cnn_batchnorm=[False, False, True, True],
            image_sequencer="none-16",
            rnn_units=128,
            rnn_layers=4,
            rnn_dropout=0.5,
            lin_dropout=0.5,
            rnn_type=torch.nn.LSTM,
        )
        x = Variable(torch.randn((5, 3, 128, 300)), requires_grad=True)
        y = m(x)
        ys = list(y.size())
        # Check output size
        self.assertListEqual(ys, [300 // 8, 5, 30])
        # Check number of parameters
        self.assertEqual(sum(p.numel() for p in m.parameters()), 2422094)
        # Check gradient
        dx, = torch.autograd.grad([y.sum()], [x])
        self.assertNotAlmostEqual(torch.sum(dx.data), 0.0)

    def test_avgpool16(self):
        m = LaiaCRNN(
            3,
            30,
            cnn_num_features=[16, 32, 48, 64],
            cnn_kernel_size=[3, 3, 3, 3],
            cnn_stride=[1, 1, 1, 1],
            cnn_dilation=[1, 1, 1, 1],
            cnn_activation=[torch.nn.ReLU] * 4,
            cnn_poolsize=[2, 2, 2, 0],
            cnn_dropout=[0, 0, 0.2, 0.1],
            cnn_batchnorm=[False, False, True, True],
            image_sequencer="avgpool-16",
            rnn_units=128,
            rnn_layers=4,
            rnn_dropout=0.5,
            lin_dropout=0.5,
            rnn_type=torch.nn.LSTM,
        )
        x = Variable(torch.randn((5, 3, 150, 300)), requires_grad=True)
        y = m(
            PaddedTensor(
                data=x,
                sizes=Variable(
                    torch.LongTensor(
                        [[128, 300], [150, 290], [70, 200], [122, 200], [16, 20]]
                    )
                ),
            )
        )
        y, ys = pad_packed_sequence(y)
        # Check output size
        self.assertListEqual(ys, [300 // 8, 290 // 8, 200 // 8, 200 // 8, 20 // 8])
        # Check number of parameters
        self.assertEqual(sum(p.numel() for p in m.parameters()), 2422094)
        # Check gradient
        dx, = torch.autograd.grad([y.sum()], [x])
        self.assertNotAlmostEqual(torch.sum(dx.data), 0.0)


if __name__ == "__main__":
    unittest.main()
