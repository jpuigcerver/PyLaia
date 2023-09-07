import unittest

import pytest
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from laia.data import PaddedTensor
from laia.models.htr import LaiaCRNN


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

    def test_get_min_valid_image_size(self):
        m = LaiaCRNN(
            1,
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
        )
        self.assertEqual(m.get_min_valid_image_size(128), 8)

    def test_exception_on_small_inputs(self):
        m = LaiaCRNN(
            1,
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
        )
        x = PaddedTensor(
            data=torch.randn(4, 1, 150, 300, requires_grad=True),
            sizes=torch.tensor([[10, 15], [3, 6], [150, 300], [5, 10]]),
        )
        self.assertRaises(ValueError, m, x)

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


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize(
    "batch_data,batch_sizes,kwargs",
    [
        (
            torch.randn(2, 1, 32, 19),
            [[32, 19], [32, 13]],
            {
                "num_input_channels": 1,
                "num_output_labels": 12,
                "cnn_num_features": [16, 32, 48, 64],
                "cnn_kernel_size": [3, 3, 3, 3],
                "cnn_stride": [1, 1, 1, 1],
                "cnn_dilation": [1, 1, 1, 1],
                "cnn_activation": [torch.nn.ReLU] * 4,
                "cnn_poolsize": [2, 2, 2, 0],
                "cnn_dropout": [0, 0, 0.2, 0.1],
                "cnn_batchnorm": [False, False, True, True],
                "image_sequencer": "none-4",
                "rnn_units": 128,
                "rnn_layers": 4,
                "rnn_dropout": 0.5,
                "lin_dropout": 0.5,
                "rnn_type": torch.nn.LSTM,
            },
        ),
        (
            torch.randn(2, 1, 17, 19),
            [[13, 19], [17, 13]],
            {
                "num_input_channels": 1,
                "num_output_labels": 12,
                "cnn_num_features": [16, 32, 48, 64],
                "cnn_kernel_size": [3, 3, 3, 3],
                "cnn_stride": [1, 1, 1, 1],
                "cnn_dilation": [1, 1, 1, 1],
                "cnn_activation": [torch.nn.ReLU] * 4,
                "cnn_poolsize": [2, 2, 2, 0],
                "cnn_dropout": [0, 0, 0.2, 0.1],
                "cnn_batchnorm": [False, False, True, True],
                "image_sequencer": "avgpool-16",
                "rnn_units": 128,
                "rnn_layers": 4,
                "rnn_dropout": 0.5,
                "lin_dropout": 0.5,
                "rnn_type": torch.nn.LSTM,
            },
        ),
        (
            torch.randn(2, 1, 17, 19),
            [[13, 19], [17, 13]],
            {
                "num_input_channels": 1,
                "num_output_labels": 12,
                "cnn_num_features": [16, 32, 48, 64],
                "cnn_kernel_size": [3, 3, 3, 3],
                "cnn_stride": [1, 1, 1, 1],
                "cnn_dilation": [1, 1, 1, 1],
                "cnn_activation": [torch.nn.ReLU] * 4,
                "cnn_poolsize": [2, 2, 2, 0],
                "cnn_dropout": [0, 0, 0.2, 0.1],
                "cnn_batchnorm": [False, False, True, True],
                "image_sequencer": "maxpool-8",
                "rnn_units": 128,
                "rnn_layers": 4,
                "rnn_dropout": 0.5,
                "lin_dropout": 0.5,
                "rnn_type": torch.nn.LSTM,
            },
        ),
        (
            torch.randn(2, 1, 17, 19),
            [[13, 19], [17, 13]],
            {
                "num_input_channels": 1,
                "num_output_labels": 12,
                "cnn_num_features": [16, 32, 48, 64],
                "cnn_kernel_size": [3, 3, 3, 3],
                "cnn_stride": [1, 1, 1, 1],
                "cnn_dilation": [1, 1, 1, 1],
                "cnn_activation": [torch.nn.ReLU] * 4,
                "cnn_poolsize": [2, 2, 2, 0],
                "cnn_dropout": [0, 0, 0, 0],
                "cnn_batchnorm": [False, False, True, True],
                "image_sequencer": "avgpool-8",
                "rnn_units": 128,
                "rnn_layers": 4,
                "rnn_dropout": 0,
                "lin_dropout": 0,
                "rnn_type": torch.nn.LSTM,
            },
        ),
    ],
)
def test_backprop(dtype, device, batch_data, batch_sizes, kwargs):
    # Note: this only checks that the gradient w.r.t. all layers is different from zero.
    m = LaiaCRNN(**kwargs).to(device, dtype=dtype).train()
    # Convert batch input and batch sizes to appropriate type
    x = batch_data.to(device=device, dtype=dtype)
    xs = torch.tensor(batch_sizes, device=device)

    # Check model for normal tensor inputs
    m.zero_grad()
    cost = m(x).sum()
    cost.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, f"Parameter {n} does not have a gradient"
        sp = torch.abs(p.grad).sum()
        assert not torch.allclose(
            sp, torch.tensor(0, dtype=dtype)
        ), f"Gradients for parameter {n} are close to 0 ({sp:g})"

    # Check model for padded tensor inputs
    m.zero_grad()
    cost = cost_function(m(PaddedTensor(x, xs)))
    cost.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, f"Parameter {n} does not have a gradient"
        sp = torch.abs(p.grad).sum()
        assert not torch.allclose(
            sp, torch.tensor(0, dtype=dtype)
        ), f"Gradients for parameter {n} are close to 0 ({sp:g})"


if __name__ == "__main__":
    unittest.main()
