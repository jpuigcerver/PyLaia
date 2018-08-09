from __future__ import absolute_import

import unittest

import numpy as np
from PIL import Image

from laia.data.transformers.transformer_image_convert import TransformerImageConvert


class TransformerImageConvertTest(unittest.TestCase):
    def test_greyscale(self):
        t = TransformerImageConvert(mode="L")
        x = Image.new("RGB", (30, 40), color=(127, 127, 127))
        y = t(x)
        self.assertTupleEqual(x.size, y.size)
        self.assertEqual("L", y.mode)
        y = np.asarray(y)
        self.assertEqual(y.max(), 127)
        self.assertEqual(y.min(), 127)

    def test_rgba(self):
        t = TransformerImageConvert(mode="RGBA")
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
