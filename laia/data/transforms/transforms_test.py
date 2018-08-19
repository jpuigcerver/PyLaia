from __future__ import absolute_import

import unittest

from PIL import Image
import numpy as np

from laia.data.transforms.transforms import RandomProbChoice, Invert, Convert


def fa(x):
    return x + 1


def fb(x):
    return x - 1


def fc(x):
    return x * 2


class TransformerTest(unittest.TestCase):
    def test_prob_choice(self):
        # Equally probable choice
        t = RandomProbChoice((fa, fb))
        self.assertSetEqual({2, 4}, set([t(3) for _ in range(50)]))
        # Choices with probabilities
        t = RandomProbChoice(((0.0, fa), (1.0, fb), (0.0, fb)))
        self.assertListEqual([2] * 50, [t(3) for _ in range(50)])
        t = RandomProbChoice(((0.5, fa), (0.5, fb)))
        self.assertSetEqual({2, 4}, set([t(3) for _ in range(50)]))

    def test_invert(self):
        t = Invert()
        x = Image.new("L", (30, 40), color=0)
        y = t(x)
        self.assertTupleEqual(x.size, y.size)
        self.assertEqual(x.mode, y.mode)
        y = np.asarray(y)
        self.assertEqual(y.max(), 255)
        self.assertEqual(y.min(), 255)

    def test_convert_greyscale(self):
        t = Convert(mode="L")
        x = Image.new("RGB", (30, 40), color=(127, 127, 127))
        y = t(x)
        self.assertTupleEqual(x.size, y.size)
        self.assertEqual("L", y.mode)
        y = np.asarray(y)
        self.assertEqual(y.max(), 127)
        self.assertEqual(y.min(), 127)

    def test_convert_rgba(self):
        t = Convert(mode="RGBA")
        x = Image.new("L", (30, 40), color=127)
        y = t(x)
        self.assertTupleEqual(x.size, y.size)
        self.assertEqual("RGBA", y.mode)
        y = np.asarray(y)
        # Alpha channel
        self.assertEqual(y[:, :, -1].max(), 255)
        self.assertEqual(y[:, :, -1].min(), 255)
        # Color channels
        self.assertEqual(y[:, :, :3].max(), 127)
        self.assertEqual(y[:, :, :3].min(), 127)
