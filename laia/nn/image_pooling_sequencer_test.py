import unittest

import torch
from laia.nn.image_pooling_sequencer import ImagePoolingSequencer


class ImagePoolingSequencerTest(unittest.TestCase):
    def test_bad_sequencer(self):
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="foo")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="maxpool")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="avgpool-")
        self.assertRaises(ValueError, ImagePoolingSequencer, sequencer="maxpool-c")


def _generate_test(sequencer, poolsize, columnwise, x, output_size):
    def _test(self):
        m = ImagePoolingSequencer(
            sequencer="{}-{}".format(sequencer, poolsize), columnwise=columnwise
        )
        self.assertListEqual(list(m(x).size()), output_size)

    return _test


def _generate_failing_test(sequencer, poolsize, columnwise, x):
    def _test(self):
        m = ImagePoolingSequencer(
            sequencer="{}-{}".format(sequencer, poolsize), columnwise=columnwise
        )

        def wrap_call():
            return m(x)

        self.assertRaises(ValueError, wrap_call)

    return _test


for sequencer in ["none", "maxpool", "avgpool"]:
    setattr(
        ImagePoolingSequencerTest,
        "test_tensor_{}_col".format(sequencer),
        _generate_test(
            sequencer=sequencer,
            poolsize=10,
            columnwise=True,
            x=torch.randn(2, 3, 10, 11),
            output_size=[11, 2, 3 * 10],
        ),
    )
    setattr(
        ImagePoolingSequencerTest,
        "test_tensor_{}_row".format(sequencer),
        _generate_test(
            sequencer=sequencer,
            poolsize=11,
            columnwise=False,
            x=torch.randn(2, 3, 10, 11),
            output_size=[10, 2, 3 * 11],
        ),
    )

for columnwise in [True, False]:
    setattr(
        ImagePoolingSequencerTest,
        "test_tensor_bad_input_{}".format("col" if columnwise else "row"),
        _generate_failing_test(
            sequencer="none",
            poolsize=9,
            columnwise=columnwise,
            x=torch.randn(2, 3, 4, 5),
        ),
    )

if __name__ == "__main__":
    unittest.main()
