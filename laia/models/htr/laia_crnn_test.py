import unittest

import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from laia.data import PaddedTensor
from laia.models.htr.laia_crnn import LaiaCRNN
from laia.models.htr.testing_utils import generate_backprop_floating_point_tests

try:
    import nnutils_pytorch

    nnutils_installed = True
except ImportError:
    nnutils_installed = False


class LaiaCRNNTest(unittest.TestCase):
    def test_get_conv_output_size(self):
        ys = LaiaCRNN.get_conv_output_size(
            size=(30, 40),
            cnn_kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
            cnn_stride=[(1, 1), (1, 1), (1, 1), (1, 1)],
            cnn_dilation=[(1, 1), (1, 1), (1, 1), (1, 1)],
            cnn_poolsize=[(2, 2), (2, 2), (2, 2), (1, 1)],
        )
        self.assertEqual(ys, (30 // 8, 40 // 8))

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
        x = torch.randn(5, 3, 128, 300, requires_grad=True)
        y = m(x)
        # Check output size
        self.assertEqual([300 // 8, 5, 30], list(y.size()))
        # Check number of parameters
        self.assertEqual(2421982, sum(p.numel() for p in m.parameters()))
        # Check gradient
        (dx,) = torch.autograd.grad([y.sum()], [x])
        self.assertNotAlmostEqual(0.0, torch.sum(dx).item())

    @unittest.skipIf(not nnutils_installed, "nnutils does not seem installed")
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
        x = torch.randn(5, 3, 150, 300, requires_grad=True)
        y = m(
            PaddedTensor(
                data=x,
                sizes=torch.tensor(
                    [[128, 300], [150, 290], [70, 200], [122, 200], [16, 20]]
                ),
            )
        )
        y, ys = pad_packed_sequence(y)
        # Check output size
        self.assertEqual(ys.tolist(), [300 // 8, 290 // 8, 200 // 8, 200 // 8, 20 // 8])
        # Check number of parameters
        self.assertEqual(2421982, sum(p.numel() for p in m.parameters()))
        # Check gradient
        (dx,) = torch.autograd.grad([y.sum()], [x])
        self.assertNotAlmostEqual(0.0, torch.sum(dx).item())


def cost_function(y):
    assert isinstance(y, (torch.Tensor, PackedSequence))
    if isinstance(y, PackedSequence):
        y = y.data
    return y.sum()


# Add some tests to make sure that the backprop is working correctly.
# Note: this only checks that the gradient w.r.t. all layers is different from zero.
tests = [
    (
        "backprop_{}_{}_fixed_size",
        dict(
            module=LaiaCRNN,
            module_kwargs=dict(
                num_input_channels=1,
                num_output_labels=12,
                cnn_num_features=[16, 32, 48, 64],
                cnn_kernel_size=[3, 3, 3, 3],
                cnn_stride=[1, 1, 1, 1],
                cnn_dilation=[1, 1, 1, 1],
                cnn_activation=[torch.nn.ReLU] * 4,
                cnn_poolsize=[2, 2, 2, 0],
                cnn_dropout=[0, 0, 0.2, 0.1],
                cnn_batchnorm=[False, False, True, True],
                image_sequencer="none-4",
                rnn_units=128,
                rnn_layers=4,
                rnn_dropout=0.5,
                lin_dropout=0.5,
                rnn_type=torch.nn.LSTM,
            ),
            batch_data=torch.randn(2, 1, 32, 19),
            batch_sizes=[[32, 19], [32, 13]],
            cost_function=cost_function,
            padded_cost_function=cost_function,
        ),
    )
]

if nnutils_installed:
    tests += [
        (
            "backprop_{}_{}_avgpool",
            dict(
                module=LaiaCRNN,
                module_kwargs=dict(
                    num_input_channels=1,
                    num_output_labels=12,
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
                ),
                batch_data=torch.randn(2, 1, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=cost_function,
            ),
        ),
        (
            "backprop_{}_{}_maxpool",
            dict(
                module=LaiaCRNN,
                module_kwargs=dict(
                    num_input_channels=1,
                    num_output_labels=12,
                    cnn_num_features=[16, 32, 48, 64],
                    cnn_kernel_size=[3, 3, 3, 3],
                    cnn_stride=[1, 1, 1, 1],
                    cnn_dilation=[1, 1, 1, 1],
                    cnn_activation=[torch.nn.ReLU] * 4,
                    cnn_poolsize=[2, 2, 2, 0],
                    cnn_dropout=[0, 0, 0.2, 0.1],
                    cnn_batchnorm=[False, False, True, True],
                    image_sequencer="maxpool-8",
                    rnn_units=128,
                    rnn_layers=4,
                    rnn_dropout=0.5,
                    lin_dropout=0.5,
                    rnn_type=torch.nn.LSTM,
                ),
                batch_data=torch.randn(2, 1, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=cost_function,
            ),
        ),
        (
            "backprop_{}_{}_no_dropout",
            dict(
                module=LaiaCRNN,
                module_kwargs=dict(
                    num_input_channels=1,
                    num_output_labels=12,
                    cnn_num_features=[16, 32, 48, 64],
                    cnn_kernel_size=[3, 3, 3, 3],
                    cnn_stride=[1, 1, 1, 1],
                    cnn_dilation=[1, 1, 1, 1],
                    cnn_activation=[torch.nn.ReLU] * 4,
                    cnn_poolsize=[2, 2, 2, 0],
                    cnn_dropout=[0, 0, 0, 0],
                    cnn_batchnorm=[False, False, True, True],
                    image_sequencer="avgpool-8",
                    rnn_units=128,
                    rnn_layers=4,
                    rnn_dropout=0,
                    lin_dropout=0,
                    rnn_type=torch.nn.LSTM,
                ),
                batch_data=torch.randn(2, 1, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=cost_function,
            ),
        ),
    ]

generate_backprop_floating_point_tests(LaiaCRNNTest, tests=tests)

if __name__ == "__main__":
    unittest.main()
