from __future__ import absolute_import

import unittest

import numpy as np
from PIL import Image

from laia.data.transformers.transformer_image_invert import TransformerImageInvert


class TransformerImageInvertTest(unittest.TestCase):
    def test(self):
        t = TransformerImageInvert()
        x = Image.new("L", (30, 40), color=0)
        y = t(x)
        self.assertTupleEqual(x.size, y.size)
        self.assertEqual(x.mode, y.mode)
        y = np.asarray(y)
        self.assertEqual(y.max(), 255)
        self.assertEqual(y.min(), 255)
