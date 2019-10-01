from typing import Callable, Union, Tuple, Sequence, List

import numpy as np
import torchvision


class RandomProbChoice(torchvision.transforms.transforms.RandomTransforms):
    """Apply a randomly transformation chosen from a given set with some probability."""

    def __init__(
        self, transforms: Sequence[Union[Callable, Tuple[float, Callable]]]
    ) -> None:
        assert transforms, "You must specify at least one choice"

        callables = []
        self._probs = []
        for transformer in transforms:
            if isinstance(transformer, tuple):
                self._probs.append(transformer[0])
                callables.append(transformer[1])
            else:
                callables.append(transformer)
        if self._probs:
            assert len(self._probs) == len(callables)
        else:
            self._probs = None

        super(RandomProbChoice, self).__init__(callables)

    def __call__(self, x):
        t = np.random.choice(np.arange(len(self.transforms)), p=self._probs)
        return self.transforms[t](x)


class Identity(object):
    def __call__(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


Compose = torchvision.transforms.transforms.Compose
RandomApply = torchvision.transforms.transforms.RandomApply
RandomChoice = torchvision.transforms.transforms.RandomChoice
