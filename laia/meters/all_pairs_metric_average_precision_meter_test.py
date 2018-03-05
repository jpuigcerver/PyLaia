from __future__ import absolute_import

from laia.meters import AllPairsMetricAveragePrecisionMeter

import unittest


class AllPairsMetricAveragePrecisionMeterTest(unittest.TestCase):
    def setUp(self):
        self.batch1 = ([[1, 1],
                        [2, 2]],
                       [1, 2])
        self.batch2 = ([[0, 0],
                        [1, 1],
                        [0, 2]],
                       [3, 1, 2])

    def test(self):
        meter = AllPairsMetricAveragePrecisionMeter()
        # Add batches to the meter
        meter.add(*self.batch1)
        meter.add(*self.batch2)

        g_ap, m_ap = meter.value
        self.assertEqual(8.0 / 12.0, g_ap)
        self.assertEqual(8.0 / 12.0, m_ap)

    def test_with_singletons(self):
        meter = AllPairsMetricAveragePrecisionMeter(
            ignore_singleton=False)

        # Add batches to the meter
        meter.add(*self.batch1)
        meter.add(*self.batch2)

        g_ap, m_ap = meter.value
        self.assertEqual(0.625, g_ap)
        self.assertEqual((1.0 + 1.0/3.0 + 1.0 + 1.0/3.0 + 0.0) / 5, m_ap)


if __name__ == '__main__':
    unittest.main()
