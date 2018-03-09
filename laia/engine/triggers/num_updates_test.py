from __future__ import absolute_import

import unittest

from laia.engine.trainer import Trainer
from laia.engine.triggers.num_updates import NumUpdates


class FakeTrainer(Trainer):
    def __init__(self):
        super(FakeTrainer, self).__init__(None, None, None, None)

    def set_num_updates(self, num):
        self._updates = num


class NumUpdatesTest(unittest.TestCase):
    def test_below(self):
        trainer = FakeTrainer()
        trigger = NumUpdates(trainer, 25, 'TriggerName')
        self.assertEqual(False, trigger())

    def test_above(self):
        trainer = FakeTrainer()
        trainer.set_num_updates(25)
        trigger = NumUpdates(trainer, 25, 'TriggerName2')
        self.assertEqual(True, trigger())


if __name__ == '__main__':
    unittest.main()
