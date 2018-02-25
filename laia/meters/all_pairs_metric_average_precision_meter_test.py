from __future__ import absolute_import

from laia.meters import AllPairsMetricAveragePrecisionMeter

import torch
import unittest


def features_fn(batch):
    return batch[0]

def class_fn(batch):
    return batch[1]

class AllPairsMetricAveragePrecisionMeterTest(unittest.TestCase):
    def setUp(self):
        self.batch1 = (torch.Tensor([[1, 1],
                                     [2, 2]]),
                       [1, 2])
        self.batch2 = (torch.Tensor([[1, 1],
                                     [0, 2],
                                     [0, 0]]),
                       [1, 2, 3])

    def test(self):
        meter = AllPairsMetricAveragePrecisionMeter(
            features_fn=features_fn, class_fn=class_fn)

        # Add batches to the meter
        meter(batch=self.batch1)
        meter(batch=self.batch2)

        g_ap, m_ap = meter.value
        self.assertEqual(8.0 / 12.0, g_ap)
        self.assertEqual(8.0 / 12.0, m_ap)

    def test_with_singletons(self):
        meter = AllPairsMetricAveragePrecisionMeter(
            features_fn=features_fn, class_fn=class_fn,
            ignore_singleton=False)

        # Add batches to the meter
        meter(batch=self.batch1)
        meter(batch=self.batch2)

        g_ap, m_ap = meter.value
        self.assertEqual(0.625, g_ap)
        self.assertEqual((1.0 + 1.0/3.0 + 1.0 + 1.0/3.0 + 0.0) / 5, m_ap)


if __name__ == '__main__':
    unittest.main()
