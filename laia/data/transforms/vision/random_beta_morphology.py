from typing import List, Tuple, Union

import numpy as np
import scipy.special
from PIL import Image, ImageFilter


class RandomBetaMorphology:
    def __init__(
        self, filter_size_min: int, filter_size_max: int, alpha: float, beta: float
    ) -> None:
        assert filter_size_min % 2 != 0, "Filter size must be odd"
        assert filter_size_max % 2 != 0, "Filter size must be odd"
        self.filter_size_min = filter_size_min
        self.filter_size_max = filter_size_max
        self.alpha = alpha
        self.beta = beta
        self.filter_sizes, self.filter_probs = self._create_filter_distribution(
            filter_size_min, filter_size_max, alpha, beta
        )

    @staticmethod
    def _create_filter_distribution(
        filter_size_min: int, filter_size_max: int, alpha: float, beta: float
    ) -> Tuple[List[int], Union[List[float], np.ndarray]]:
        n = (filter_size_max - filter_size_min) // 2 + 1
        if n < 2:
            return [filter_size_min], np.asarray([1.0], dtype=np.float32)
        filter_sizes = []
        filter_probs = []
        for k in range(n):
            filter_sizes.append(filter_size_min + 2 * k)
            filter_probs.append(
                scipy.special.comb(n, k) * scipy.special.beta(alpha + k, n - k + beta)
            )
        np_filter_probs = np.asarray(filter_probs, dtype=np.float32)
        np_filter_probs = filter_probs / np_filter_probs.sum()
        return filter_sizes, np_filter_probs

    def sample_filter_size(self):
        filter_size = np.random.choice(self.filter_sizes, p=self.filter_probs)
        return filter_size

    def __call__(self, *args, **kwargs):
        return NotImplementedError

    def __repr__(self) -> str:
        return (
            f"vision.{self.__class__.__name__}("
            f"filter_size_min={self.filter_size_min}, "
            f"filter_size_max={self.filter_size_max}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )


class Dilate(RandomBetaMorphology):
    def __init__(
        self,
        filter_size_min: int = 3,
        filter_size_max: int = 7,
        alpha: float = 1,
        beta: float = 3,
    ) -> None:
        super().__init__(filter_size_min, filter_size_max, alpha, beta)

    def __call__(self, img: Image) -> Image:
        filter_size = self.sample_filter_size()
        return img.filter(ImageFilter.MaxFilter(filter_size))


class Erode(RandomBetaMorphology):
    def __init__(
        self,
        filter_size_min: int = 3,
        filter_size_max: int = 5,
        alpha: float = 1,
        beta: float = 3,
    ) -> None:
        super().__init__(filter_size_min, filter_size_max, alpha, beta)

    def __call__(self, img: Image) -> Image:
        filter_size = self.sample_filter_size()
        return img.filter(ImageFilter.MinFilter(filter_size))


if __name__ == "__main__":
    import argparse

    from PIL import ImageOps

    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("dilate", "erode"), default="dilate")
    parser.add_argument("images", type=argparse.FileType("rb"), nargs="+")
    args = parser.parse_args()

    transformer = Dilate() if args.operation == "dilate" else Erode()

    for f in args.images:
        x = Image.open(f, "r").convert("L")
        x = ImageOps.invert(x)
        y = transformer(x)

        w, h = x.size
        z = Image.new("L", (w, 2 * h))
        z.paste(x, (0, 0))
        z.paste(y, (0, h))
        z = z.resize(size=(w // 2, h), resample=Image.BICUBIC)
        z.show()
        input()
