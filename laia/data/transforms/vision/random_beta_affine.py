from typing import Tuple, Union

import numpy as np
from PIL import Image
from scipy.linalg import solve


class RandomBetaAffine:
    """Apply a random affine transform on a PIL image
    using a Beta distribution."""

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

    def __call__(self, img: Image.Image) -> Image.Image:
        max_offset = min(img.size) * self.max_offset_ratio
        z = np.random.beta(self.alpha, self.beta, size=(3, 2))
        offset = ((2.0 * z - 1.0) * max_offset).astype(np.float32)
        w, h = img.size
        src = np.asarray([(0, 0), (0, h), (w, 0)], dtype=np.float32)
        dst = src + offset
        affine_mat = self.get_affine_transform(src, dst)
        return img.transform(
            img.size,
            method=Image.AFFINE,
            data=affine_mat,
            resample=Image.BILINEAR,
            fillcolor=self.fillcolor,
        )

    def __repr__(self) -> str:
        return (
            f"vision.{self.__class__.__name__}("
            f"max_offset_ratio={self.max_offset_ratio}, "
            f"alpha={self.alpha}, beta={self.beta}"
            f"{f', fillcolor={self.fillcolor}' if self.fillcolor else ''})"
        )

    @staticmethod
    def get_affine_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
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
    parser.add_argument("--max_offset_ratio", type=float, default=0.2)
    parser.add_argument("images", type=argparse.FileType("rb"), nargs="+")
    args = parser.parse_args()

    transformer = RandomBetaAffine(
        alpha=args.alpha, beta=args.beta, max_offset_ratio=args.max_offset_ratio
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
