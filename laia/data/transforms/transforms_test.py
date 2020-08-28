import unittest

from laia.data.transforms import RandomProbChoice


def fa(x):
    return x + 1


def fb(x):
    return x - 1


class TransformerTest(unittest.TestCase):
    def test_prob_choice(self):
        # Equally probable choice
        t = RandomProbChoice((fa, fb))
        self.assertSetEqual({2, 4}, set([t(3) for _ in range(50)]))
        # Choices with probabilities
        t = RandomProbChoice(((0.0, fa), (1.0, fb), (0.0, fb)))
        self.assertListEqual([2] * 50, [t(3) for _ in range(50)])
        t = RandomProbChoice(((0.5, fa), (0.5, fb)))
        self.assertSetEqual({2, 4}, set([t(3) for _ in range(50)]))
