from __future__ import absolute_import

import unittest

import torch

from laia.models.htr.dortmund_crnn import DortmundCRNN
from laia.models.htr.testing_utils import generate_backprop_floating_point_tests


class DortmundCRNNTest(unittest.TestCase):
    pass


# Add some tests to make sure that the backprop is working correctly.
# Note: this only checks that the gradient w.r.t. all layers is different from zero.
test_dict = {
    "module": DortmundCRNN,
    "batch_data": torch.randn(2, 1, 17, 19),
    "batch_sizes": [[13, 19], [17, 13]],
    "cost_function": lambda y: y.sum(),
    "padded_cost_function": lambda y: y.sum(),
}
tests = []
for name, kwargs in (("default", {}), ("no_dropout", {"dropout": 0})):
    test_dict["module_kwargs"] = dict(num_outputs=36, **kwargs)
    tests.append(("backprop_{{}}_{{}}_{}".format(name), test_dict))
generate_backprop_floating_point_tests(DortmundCRNNTest, tests=tests)

if __name__ == "__main__":
    unittest.main()
