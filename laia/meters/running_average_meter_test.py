from __future__ import absolute_import
from __future__ import division

import math
import unittest

from laia.meters.running_average_meter import RunningAverageMeter


class RunningAverageMeterTest(unittest.TestCase):
    def testMeter(self):
        m = RunningAverageMeter()
        m.add(25)
        self.assertEquals(m.value, (25.0, 0.0))
        m.add(25)
        self.assertEquals(m.value, (25.0, 0.0))
        m.add(70)
        stddev = math.sqrt((25.0 ** 2 + 25.0 ** 2 + 70.0 ** 2) / 3.0 - 1600.0)
        self.assertEquals(m.value, (40.0, stddev))
        m.add(-120)
        stddev = math.sqrt((25.0 ** 2 + 25.0 ** 2 + 70.0 ** 2 + 120 ** 2) / 4.0)
        self.assertEquals(m.value, (0.0, stddev))


if __name__ == '__main__':
    unittest.main()
