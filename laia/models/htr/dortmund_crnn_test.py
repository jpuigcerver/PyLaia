from __future__ import absolute_import

import unittest
import torch

from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from laia.models.htr.dortmund_crnn import DortmundCRNN
from laia.models.htr.testing_utils import generate_backprop_floating_point_tests


class DortmundCRNNTest(unittest.TestCase):
    pass


def cost_function(y):
    assert isinstance(y, (Variable, PackedSequence))
    y = y if isinstance(y, Variable) else y.data
    return y.sum()


# Add some tests to make sure that the backprop is working correctly.
# Note: this only checks that the gradient w.r.t. all layers is different from zero.
generate_backprop_floating_point_tests(
    DortmundCRNNTest,
    tests=[
        (
            "backprob_{}_{}_default",
            dict(
                module=DortmundCRNN,
                module_kwargs=dict(num_outputs=36),
                batch_data=torch.randn(2, 1, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=cost_function,
            ),
        ),
        (
            "backprob_{}_{}_no_dropout",
            dict(
                module=DortmundCRNN,
                module_kwargs=dict(num_outputs=36, dropout=0),
                batch_data=torch.randn(2, 1, 17, 19),
                batch_sizes=[[13, 19], [17, 13]],
                cost_function=cost_function,
                padded_cost_function=cost_function,
            ),
        ),
    ],
)

if __name__ == "__main__":
    unittest.main()
