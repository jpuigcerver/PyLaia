from __future__ import absolute_import

import unittest
from laia.data.transformers.transformer import (
    TransformerChoice,
    TransformerConditional,
    TransformerList,
)


def fa(x):
    return x + 1


def fb(x):
    return x - 1


def fc(x):
    return x * 2


class TransformerTest(unittest.TestCase):
    def test_conditional(self):
        t = TransformerConditional(fa, p=1.0)
        self.assertEqual(4, t(3))
        t = TransformerConditional(fa, p=0.0)
        self.assertEqual(3, t(3))

    def test_choice(self):
        # Only one choice
        t = TransformerChoice(fa)
        self.assertEqual(4, t(3))
        # Equally probable choice
        t = TransformerChoice(fa, fb)
        self.assertSetEqual({2, 4}, set([t(3) for _ in range(50)]))
        # Choices with probabilities
        t = TransformerChoice((0.0, fa), (1.0, fb), (0.0, fb))
        self.assertListEqual([2] * 50, [t(3) for _ in range(50)])
        t = TransformerChoice((0.5, fa), (0.5, fb))
        self.assertSetEqual({2, 4}, set([t(3) for _ in range(50)]))

    def test_sequence(self):
        t = TransformerList(fa, fc, fb)
        self.assertEqual(7, t(3))
