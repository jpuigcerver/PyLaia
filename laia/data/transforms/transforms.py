from typing import Callable, Sequence, Tuple, Union

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
        for t in transforms:
            if isinstance(t, tuple):
                p, c = t
                self._probs.append(p)
                callables.append(c)
            else:
                callables.append(t)
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
