from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from scipy.linalg import solve
from typing import Union, Tuple

from laia.data.transformers.transformer import Transformer


class TransformerImageAffine(Transformer):
    """Apply a random affine transform on a PIL image."""

    def __init__(
        self,
        max_offset_ratio=0.2,  # type: float
        alpha=2,  # type: float
        beta=2,  # type: float
        fillcolor=None,  # type: Union[None, int, Tuple[int, int, int]]
    ):
        assert max_offset_ratio > 0
        assert alpha > 0
        assert beta > 0
        super(Transformer, self).__init__()
        self.max_offset_ratio = max_offset_ratio
        self.alpha = alpha
        self.beta = beta
        self.fillcolor = fillcolor

    def __call__(self, x):
        # type: (Image.Image) -> Image.Image
        max_offset = min(x.size) * self.max_offset_ratio
        z = np.random.beta(self.alpha, self.beta, size=(3, 2))
        offset = ((2.0 * z - 1.0) * max_offset).astype(np.float32)
        w, h = x.size
        src = np.asarray([(0, 0), (0, h), (w, 0)], dtype=np.float32)
        dst = src + offset
        affine_mat = self.get_affine_transform(src, dst)
        return x.transform(
            x.size,
            method=Image.AFFINE,
            data=affine_mat,
            resample=Image.BILINEAR,
            fillcolor=self.fillcolor,
        )

    def _to_string(self, spaces):
        return (
            (" " * spaces)
            + self._type()
            + "(max_offset_ratio=%g, alpha=%g, beta=%g)"
            % (self.max_offset_ratio, self.alpha, self.beta)
        )

    @staticmethod
    def get_affine_transform(src, dst):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        assert src.shape == (3, 2)
        assert dst.shape == (3, 2)
        coeffs = np.zeros((6, 6), dtype=np.float32)
        for i in [0, 1, 2]:
            coeffs[i, 0:2] = coeffs[i + 3, 3:5] = src[i]
            coeffs[i, 2] = coeffs[i + 3, 5] = 1
        return solve(coeffs, dst.transpose().flatten())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=2)
    parser.add_argument("--max_offset_ratio", type=float, default=0.3)
    parser.add_argument("image", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    transformer = TransformerImageAffine(
        alpha=args.alpha, beta=args.beta, max_offset_ratio=args.max_offset_ratio
    )
    print(transformer)
    for f in args.image:
        x = Image.open(f, "r").convert("L")
        y = transformer(x)

        w, h = x.size
        z = Image.new("L", (w, 2 * h))
        z.paste(x, (0, 0))
        z.paste(y, (0, h))
        z = z.resize(size=(w // 2, h), resample=Image.BICUBIC)
        z.show()
        try:
            raw_input()
        except NameError:
            input()
