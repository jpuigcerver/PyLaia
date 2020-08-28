import unittest

import torch

from laia.data import PaddedTensor
from laia.models.htr import ConvBlock
from laia.models.htr.testing_utils import generate_backprop_floating_point_tests


class ConvBlockTest(unittest.TestCase):
    def test_output_size(self):
        m = ConvBlock(4, 5, kernel_size=3, stride=1, dilation=1, poolsize=2)
        x = torch.randn(3, 4, 11, 13)
        y = m(x)
        self.assertEqual((3, 5, 11 // 2, 13 // 2), tuple(y.size()))

    def test_output_size_padded_tensor(self):
        m = ConvBlock(4, 5, kernel_size=3, stride=1, dilation=1, poolsize=2)
        x = torch.randn(3, 4, 11, 13)
        y = m(PaddedTensor(x, torch.tensor([[11, 13], [10, 12], [3, 2]])))
        self.assertEqual(
            [[11 // 2, 13 // 2], [10 // 2, 12 // 2], [3 // 2, 2 // 2]], y.sizes.tolist()
        )

    def test_output_size_no_pool(self):
        m = ConvBlock(4, 5, poolsize=0)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(x, torch.tensor([[11, 13]])))
        self.assertEqual([[11, 13]], y.sizes.tolist())
        self.assertEqual([11, 13], list(y.data.size())[2:])

    def test_output_size_poolsize(self):
        m = ConvBlock(4, 5, poolsize=3)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(x, torch.tensor([[11, 13]])))
        self.assertEqual([[11 // 3, 13 // 3]], y.sizes.tolist())
        self.assertEqual([11 // 3, 13 // 3], list(y.data.size())[2:])

    def test_output_size_dilation(self):
        # Note: padding should be added automatically to have the same output size
        m = ConvBlock(4, 5, dilation=3)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(x, torch.tensor([[11, 13]])))
        self.assertEqual([[11, 13]], y.sizes.tolist())
        self.assertEqual([11, 13], list(y.data.size())[2:])

    def test_output_size_stride(self):
        m = ConvBlock(4, 5, stride=2)
        x = torch.randn(1, 4, 11, 13)
        y = m(PaddedTensor(x, torch.tensor([[11, 13]])))
        self.assertEqual([[11 // 2 + 1, 13 // 2 + 1]], y.sizes.tolist())
        self.assertEqual([11 // 2 + 1, 13 // 2 + 1], list(y.data.size())[2:])

    def test_masking(self):
        m = ConvBlock(1, 1, activation=None, use_masks=True)
        # Reset parameters so that the operation does nothing
        for name, param in m.named_parameters():
            param.data.zero_()
            if name == "conv.weight":
                param[:, :, 1, 1] = 1

        x = torch.randn(3, 1, 11, 13)
        y = m(PaddedTensor(x, torch.tensor([[11, 13], [10, 12], [3, 2]]))).data

        # Check sample 1
        torch.testing.assert_allclose(x[0, :, :, :], y[0, :, :, :])
        # Check sample 2
        torch.testing.assert_allclose(x[1, :, :10, :12], y[1, :, :10, :12])
        torch.testing.assert_allclose(torch.zeros(1, 1, 13), y[1, :, 10:, :])
        torch.testing.assert_allclose(torch.zeros(1, 11, 1), y[1, :, :, 12:])
        # Check sample 3
        torch.testing.assert_allclose(x[2, :, :3, :2], y[2, :, :3, :2])
        torch.testing.assert_allclose(torch.zeros(1, 8, 13), y[2, :, 3:, :])
        torch.testing.assert_allclose(torch.zeros(1, 11, 11), y[2, :, :, 2:])


def padded_cost_function(padded_y):
    y, ys = padded_y.data, padded_y.sizes
    cost = 0
    for y_i, ys_i in zip(y, ys):
        cost = y_i[:, : ys_i[0], : ys_i[1]].sum()
    return cost


# Add some tests to make sure that the backprop is working correctly.
# Note: this only checks that the gradient w.r.t. all layers is different from zero.
test_dict = {
    "module": ConvBlock,
    "batch_data": torch.randn(2, 3, 17, 19),
    "batch_sizes": [[13, 19], [17, 13]],
    "cost_function": lambda y: y.sum(),
    "padded_cost_function": padded_cost_function,
}
tests = []
for name, kwargs in (
    ("default", {}),
    ("batchnorm", {"batchnorm": True}),
    ("dropout", {"dropout": 0.3}),
    ("maxpool", {"poolsize": 2}),
    ("use_masks", {"use_masks": True}),
    ("inplace", {"inplace": True}),
):
    test_dict["module_kwargs"] = {
        "in_channels": 3,
        "out_channels": 5,
        **kwargs,  # type: ignore
    }
    tests.append(("backprop_{{}}_{{}}_{}".format(name), test_dict))
generate_backprop_floating_point_tests(ConvBlockTest, tests=tests)

if __name__ == "__main__":
    unittest.main()
