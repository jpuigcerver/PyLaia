from __future__ import absolute_import
from __future__ import division

import unittest

from laia.hooks.meters.average_precision_meter import AveragePrecisionMeter


class AveragePrecisionMeterTest(unittest.TestCase):
    def setUp(self):
        self.batch1 = ([[1, 1],
                        [2, 2]],
                       [1, 2])
        self.batch2 = ([[0, 0],
                        [1, 1],
                        [0, 2]],
                       [3, 1, 2])

    def test(self):
        meter = AveragePrecisionMeter()
        meter.add(1, 0, 0)
        meter.add(0, 1, 0)
        meter.add(1, 0, 0)
        meter.add(0, 0, 1)
        self.assertEqual((1.0 / 1.0 + 2.0 / 3.0) / 3.0, meter.value)

    def test_score(self):
        meter = AveragePrecisionMeter()
        meter.add(0, 0, 1, score=0)
        meter.add(1, 0, 0, score=1)
        meter.add(0, 1, 0, score=2)
        meter.add(1, 0, 0, score=3)
        self.assertEqual((1 / 1 + 2 / 3) / 3, meter.value)

    def test_score_ascend(self):
        meter = AveragePrecisionMeter(desc_sort=False)
        meter.add(1, 0, 0, score=1)
        meter.add(0, 1, 0, score=2)
        meter.add(1, 0, 0, score=3)
        meter.add(0, 0, 1, score=4)
        self.assertEqual((1 + 2 / 3) / 3, meter.value)

    def test_score_multi(self):
        meter = AveragePrecisionMeter()
        meter.add(2, 1, 0)
        meter.add(1, 1, 0)
        meter.add(0, 3, 0)
        meter.add(0, 0, 1)
        meter.add(1, 1, 1)
        self.assertEqual((2 / 3 + 3 / 5 + 4 / 10) / 6, meter.value)


if __name__ == '__main__':
    unittest.main()
