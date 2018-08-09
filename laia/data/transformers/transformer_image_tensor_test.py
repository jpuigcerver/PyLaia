from __future__ import absolute_import

import unittest

import numpy as np
from PIL import Image

from laia.data.transformers.transformer_image_tensor import TransformerImageTensor


class TransformerImageTensorTest(unittest.TestCase):
    @staticmethod
    def test_rgb():
        t = TransformerImageTensor()
        x = np.random.rand(30, 40, 3).astype(np.float32)
        im = Image.fromarray((x * 255).astype(np.uint8)).convert("RGB")
        y = t(im)
        # Note: 1 / 255 is the maximum error due to quantization
        np.testing.assert_allclose(
            y.numpy(), x.transpose((2, 0, 1)), rtol=0, atol=1.0 / 255.0
        )

    @staticmethod
    def test_greyscale():
        t = TransformerImageTensor()
        x = np.random.rand(30, 40).astype(np.float32)
        im = Image.fromarray((x * 255).astype(np.uint8)).convert("L")
        y = t(im)
        # Note: 1 / 255 is the maximum error due to quantization
        np.testing.assert_allclose(
            y.numpy(),
            x.reshape(30, 40, 1).transpose((2, 0, 1)),
            rtol=0,
            atol=1.0 / 255.0,
        )
