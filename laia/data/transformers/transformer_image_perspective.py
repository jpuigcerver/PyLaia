from __future__ import absolute_import

from typing import Union, Tuple

from laia.data.transformers.transformer import Transformer
from PIL import Image
import numpy as np


class TransformerImagePerspective(Transformer):
    def __init__(
        self,
        probability=0.5,  # type: float
        max_offset_ratio=0.5,  # type: float
        alpha=6,  # type: float
        beta=6,  # type: float
        fillcolor=None,  # type: Union[None, int, Tuple[int, int, int]]
    ):
        assert max_offset_ratio > 0
        assert alpha > 0
        assert beta > 0
        super(Transformer, self).__init__()
        self.probability = probability
        self.max_offset_ratio = max_offset_ratio
        self.alpha = alpha
        self.beta = beta
        self.fillcolor = fillcolor

    def __call__(self, x):
        # type: (Image) -> Image
        if np.random.rand() < self.probability:
            max_offset = min(x.size) * self.max_offset_ratio
            z = np.random.beta(self.alpha, self.beta, size=(4, 2))
            offset = ((2.0 * z - 1.0) * max_offset).astype(np.float32)
            w, h = x.size
            src = np.asarray([(0, 0), (0, h), (w, 0), (w, h)], dtype=np.float32)
            dst = src + offset
            perspective_transform = self.warp_perspective(src, dst)
            return x.transform(
                x.size,
                method=Image.PERSPECTIVE,
                data=perspective_transform,
                resample=Image.BILINEAR,
                fillcolor=self.fillcolor,
            )
        else:
            return x

    @staticmethod
    def warp_perspective(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probability", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=6)
    parser.add_argument("--beta", type=float, default=6)
    parser.add_argument("--max_offset_ratio", type=float, default=0.5)
    parser.add_argument("image", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    transformer = TransformerImagePerspective(
        probability=args.probability,
        max_offset_ratio=args.max_offset_ratio,
        alpha=args.alpha,
        beta=args.beta,
    )
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
