from __future__ import absolute_import

from typing import Callable, Union, Tuple, Sequence

import numpy as np
import torchvision


class RandomProbChoice(torchvision.transforms.transforms.RandomTransforms):
    """Apply a randomly transformation chosen from a given set with some probability."""

    def __init__(self, transforms):
        # type: (Sequence[Union[Callable, Tuple[float, Callable]]]) -> None
        super(RandomProbChoice, self).__init__(transforms)
        assert transforms, "You must specify at least one choice"

        self._transforms = []
        self._probs = []
        for transformer in transforms:
            if isinstance(transformer, tuple):
                self._probs.append(transformer[0])
                self._transforms.append(transformer[1])
            else:
                self._transforms.append(transformer)
        if self._probs:
            assert len(self._probs) == len(self._transforms)
        else:
            self._probs = None

    def __call__(self, x):
        t = np.random.choice(np.arange(len(self._transforms)), p=self._probs)
        return self._transforms[t](x)


Compose = torchvision.transforms.transforms.Compose
