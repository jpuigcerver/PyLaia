from __future__ import absolute_import
from __future__ import division

import math
import unittest

from laia.meters import RunningAverageMeter


class RunningAverageMeterTest(unittest.TestCase):
    def test_meter(self):
        m = RunningAverageMeter()
        m.add(25)
        self.assertEqual(m.value, (25.0, 0.0))
        m.add(25)
        self.assertEqual(m.value, (25.0, 0.0))
        m.add(70)
        stddev = math.sqrt((25.0 ** 2 + 25.0 ** 2 + 70.0 ** 2) / 3.0 - 1600.0)
        self.assertEqual(m.value, (40.0, stddev))
        m.add(-120)
        stddev = math.sqrt((25.0 ** 2 + 25.0 ** 2 + 70.0 ** 2 + 120 ** 2) / 4.0)
        self.assertEqual(m.value, (0.0, stddev))

    def test_state_dict(self):
        m = RunningAverageMeter()
        m.add([25, 25, 70])
        m2 = RunningAverageMeter()
        m2.load_state_dict(m.state_dict())
        self.assertEqual(m.value, m2.value)


if __name__ == "__main__":
    unittest.main()
