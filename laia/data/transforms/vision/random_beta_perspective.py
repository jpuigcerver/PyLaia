from typing import Tuple, Union

import numpy as np
from PIL import Image


class RandomBetaPerspective:
    def __init__(
        self,
        max_offset_ratio: float = 0.2,
        alpha: float = 2,
        beta: float = 2,
        fillcolor: Union[None, int, Tuple[int, int, int]] = None,
    ) -> None:
        assert max_offset_ratio > 0
        assert alpha > 0
        assert beta > 0
        self.max_offset_ratio = max_offset_ratio
        self.alpha = alpha
        self.beta = beta
        self.fillcolor = fillcolor

    def __call__(self, img: Image) -> Image:
        max_offset = min(img.size) * self.max_offset_ratio
        z = np.random.beta(self.alpha, self.beta, size=(4, 2))
        offset = ((2.0 * z - 1.0) * max_offset).astype(np.float32)
        w, h = img.size
        src = np.asarray([(0, 0), (0, h), (w, 0), (w, h)], dtype=np.float32)
        dst = src + offset
        perspective_transform = self.warp_perspective(src, dst)
        return img.transform(
            img.size,
            method=Image.PERSPECTIVE,
            data=perspective_transform,
            resample=Image.BILINEAR,
            fillcolor=self.fillcolor,
        )

    def __repr__(self) -> str:
        s = "vision.{name}(max_offset_ratio={max_offset_ratio}, alpha={alpha}, beta={beta}"
        if self.fillcolor:
            s += ", fillcolor={fillcolor}"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)

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
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=2)
    parser.add_argument("--max_offset_ratio", type=float, default=0.2)
    parser.add_argument("images", type=argparse.FileType("rb"), nargs="+")
    args = parser.parse_args()

    transformer = RandomBetaPerspective(
        max_offset_ratio=args.max_offset_ratio, alpha=args.alpha, beta=args.beta
    )
    print(transformer)
    for f in args.images:
        x = Image.open(f, "r").convert("L")
        y = transformer(x)

        w, h = x.size
        z = Image.new("L", (w, 2 * h))
        z.paste(x, (0, 0))
        z.paste(y, (0, h))
        z = z.resize(size=(w // 2, h), resample=Image.BICUBIC)
        z.show()
        input()
