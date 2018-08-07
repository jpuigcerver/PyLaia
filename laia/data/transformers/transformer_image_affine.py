from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from PIL import Image
from laia.data.transformers.transformer import Transformer


class TransformerImageAffine(Transformer):
    def __init__(
        self,
        scale_prob=0.5,  # type: float
        scale_prec=70,  # type: float
        shear_prob=0.5,  # type: float
        shear_prec=4,  # type: float
        rotate_prob=0.5,  # type: float
        rotate_prec=100,  # type: float
        translate_prob=0.5,  # type: float
        translate_prec=50,  # type: float
    ):
        # type: (...) -> None
        super(TransformerImageAffine, self).__init__()
        # Scale parameters. Scaling is applied at the center of the image.
        # Log-normal distribution with mean 0.
        self.scale_prob = scale_prob
        self.scale_prec = scale_prec

        # Horizontal shear parameters.
        # Von Mises distribution with mean 0.
        self.shear_prob = shear_prob
        self.shear_prec = shear_prec

        # Rotate parameters [relative to the maximum aspect ratio of the image].
        # Von Mises distribution with mean 0.
        self.rotate_prob = rotate_prob
        self.rotate_prec = rotate_prec

        # Translate parameters [relative to the size of each dimension].
        # Normal distribution with mean 0.
        self.translate_prob = translate_prob
        self.translate_prec = translate_prec

    def _sample_matrix(self, x):
        w, h = x.size
        cx, cy = 0.5 * w, 0.5 * h
        affine_mat = None

        # Sample translation matrix
        if np.random.random() < self.translate_prob:
            dx, dy = np.random.randn(2) / math.sqrt(self.translate_prec)
            affine_mat = np.asarray(
                ((1, 0, dx), (0, 1, dy), (0, 0, 1)), dtype=np.float32
            )

        # Sample rotation matrix
        if np.random.random() < self.rotate_prob:
            kappa = max(w / h, h / w) * self.rotate_prec
            a = np.random.vonmises(0, kappa)
            sa, ca = math.sin(a), math.cos(a)
            rotate_mat = np.asarray(
                (
                    (ca, -sa, -cx * ca + cy * sa + cx),
                    (sa, ca, -cx * sa - cy * ca + cy),
                    (0, 0, 1),
                ),
                dtype=np.float32,
            )
            affine_mat = (
                rotate_mat if affine_mat is None else np.matmul(affine_mat, rotate_mat)
            )

        # Sample horizontal shear matrix
        if np.random.random() < self.shear_prob:
            m = 1.0 / math.tan(np.random.vonmises(0, self.shear_prec))
            shear_mat = np.asarray(
                ((1, m, -m * cy), (0, 1, 0), (0, 0, 1)), dtype=np.float32
            )
            affine_mat = (
                shear_mat if affine_mat is None else np.matmul(affine_mat, shear_mat)
            )

        # Sample scale matrix
        if np.random.random() < self.scale_prob:
            s = math.exp(np.random.lognormal(0, 1.0 / math.sqrt(self.scale_prec)))
            scale_mat = np.asarray(
                ((s, 0, cx - s * cx), (0, s, cy - s * cy), (0, 0, 1)), dtype=np.float32
            )
            affine_mat = (
                scale_mat if affine_mat is None else np.matmul(affine_mat, scale_mat)
            )

        return affine_mat

    def __call__(self, x):
        # type: (Image.Image) -> Image.Image
        affine_mat = self._sample_matrix(x)
        if affine_mat is None:
            return x
        else:
            a, b, c, d, e, f = affine_mat[:2, :].flatten()
            return x.transform(
                x.size,
                method=Image.AFFINE,
                data=(a, b, c, d, e, f),
                resample=Image.BILINEAR,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    transformer = TransformerImageAffine()
    for f in args.image:
        print(f)
