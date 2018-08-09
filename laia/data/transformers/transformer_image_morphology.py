from __future__ import absolute_import
from __future__ import division

import numpy as np
from PIL import Image, ImageFilter

from laia.data.transformers.transformer import Transformer


class TransformerImageMorphology(Transformer):
    def __init__(
        self,
        dilate_probability=0.35,  # type: float
        erode_probability=0.15,  # type: float
        dilate_filter_max_size=7,  # type: int
        dilate_filter_rate=0.5,  # type: float
        erode_filter_max_size=3,  # type: int
        erode_filter_rate=0.1,  # type: float
    ):
        # type: (...) -> None
        assert dilate_probability >= 0.0
        assert erode_probability >= 0.0
        assert dilate_filter_max_size >= 3, "Max filter size must be >= 3"
        assert erode_filter_max_size >= 3, "Max filter size must be >= 3"
        assert dilate_filter_rate > 0
        assert erode_filter_rate > 0
        super(TransformerImageMorphology, self).__init__()
        self.dilate_probability = dilate_probability
        self.erode_probability = erode_probability
        self.dilate_filter_sizes, self.dilate_filter_probs = self._create_filter_distribution(
            3, dilate_filter_max_size, dilate_filter_rate
        )
        self.erode_filter_sizes, self.erode_filter_probs = self._create_filter_distribution(
            3, erode_filter_max_size, erode_filter_rate
        )

    @staticmethod
    def _create_filter_distribution(min_filter_size, max_filter_size, rate_filter_size):
        filter_sizes = [min_filter_size]
        filter_weights = [1.0]
        for size in range(min_filter_size + 2, max_filter_size + 1, 2):
            filter_sizes.append(size)
            filter_weights.append(filter_weights[-1] * rate_filter_size)
        filter_weights = np.asarray(filter_weights, dtype=np.float32)
        filter_weights = filter_weights / filter_weights.sum()
        return filter_sizes, filter_weights

    def __call__(self, x):
        # type: (Image) -> Image
        r = np.random.rand()
        if r < self.dilate_probability:
            filter_size = np.random.choice(
                self.dilate_filter_sizes, p=self.dilate_filter_probs
            )
            return x.filter(ImageFilter.MaxFilter(filter_size))
        elif r < self.dilate_probability + self.erode_probability:
            filter_size = np.random.choice(
                self.erode_filter_sizes, p=self.erode_filter_probs
            )
            return x.filter(ImageFilter.MinFilter(filter_size))
        else:
            return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    transformer = TransformerImageMorphology()
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
