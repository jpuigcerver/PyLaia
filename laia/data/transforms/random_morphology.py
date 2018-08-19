from __future__ import absolute_import
from __future__ import division

import numpy as np
import scipy.special
from PIL import Image, ImageFilter


class RandomMorphology(object):
    def __init__(self, filter_size_min, filter_size_max, alpha, beta):
        # type: (int, int, float, float) -> None
        assert filter_size_min % 2 != 0, "Filter size must be odd"
        assert filter_size_max % 2 != 0, "Filter size must be odd"
        self.filter_sizes, self.filter_probs = self._create_filter_distribution(
            filter_size_min, filter_size_max, alpha, beta
        )

    @staticmethod
    def _create_filter_distribution(filter_size_min, filter_size_max, alpha, beta):
        n = (filter_size_max - filter_size_min) // 2 + 1
        if n < 2:
            return [filter_size_min], np.asarray([1.0], dtype=np.float32)
        else:
            filter_sizes = []
            filter_probs = []
            for k in range(n):
                filter_sizes.append(filter_size_min + 2 * k)
                filter_probs.append(
                    scipy.special.comb(n, k)
                    * scipy.special.beta(alpha + k, n - k + beta)
                )
            filter_probs = np.asarray(filter_probs, dtype=np.float32)
            filter_probs = filter_probs / filter_probs.sum()
            return filter_sizes, filter_probs

    def sample_filter_size(self):
        filter_size = np.random.choice(self.filter_sizes, p=self.filter_probs)
        print("filter_size = {}".format(filter_size))
        return filter_size

    def __repr__(self):
        s = "{name}(filter_size_min={filter_size_min}, filter_size_max={filter_size_max}, alpha={alpha}, beta={beta})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Dilate(RandomMorphology):
    def __init__(self, filter_size_min=3, filter_size_max=7, alpha=1, beta=3):
        super(Dilate, self).__init__(filter_size_min, filter_size_max, alpha, beta)

    def __call__(self, img):
        # type: (Image) -> Image
        filter_size = self.sample_filter_size()
        return img.filter(ImageFilter.MaxFilter(filter_size))


class Erode(RandomMorphology):
    def __init__(self, filter_size_min=3, filter_size_max=5, alpha=1, beta=3):
        super(Erode, self).__init__(filter_size_min, filter_size_max, alpha, beta)

    def __call__(self, img):
        # type: (Image) -> Image
        filter_size = self.sample_filter_size()
        return img.filter(ImageFilter.MinFilter(filter_size))


if __name__ == "__main__":
    import argparse
    from PIL import ImageOps

    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("dilate", "erode"), default="dilate")
    parser.add_argument("images", type=argparse.FileType("rb"), nargs="+")
    args = parser.parse_args()

    if args.operation == "dilate":
        transformer = Dilate()
    else:
        transformer = Erode()

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
        try:
            raw_input()
        except NameError:
            input()
