from __future__ import absolute_import

import unittest

from laia.engine.trainer import Trainer
from laia.engine.triggers.num_iterations import NumIterations


class FakeTrainer(Trainer):
    def __init__(self):
        super(FakeTrainer, self).__init__(None, None, None, None)

    def set_num_iterations(self, num):
        self._iterations = num


class NumIterationsTest(unittest.TestCase):
    def test_below(self):
        trainer = FakeTrainer()
        trigger = NumIterations(trainer, 25, 'TriggerName')
        self.assertEqual(False, trigger())

    def test_above(self):
        trainer = FakeTrainer()
        trainer.set_num_iterations(25)
        trigger = NumIterations(trainer, 25, 'TriggerName2')
        self.assertEqual(True, trigger())


if __name__ == '__main__':
    unittest.main()
