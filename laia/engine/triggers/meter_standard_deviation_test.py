from __future__ import absolute_import

import unittest

from laia.engine.triggers.meter_standard_deviation import MeterStandardDeviation
from laia.meters.meter import Meter


class ExceptionMeter(Meter):
    @property
    def value(self):
        raise Exception


class MockMeter(Meter):
    def __init__(self):
        self._value = None

    def set_value(self, value, key=None):
        if key:
            self._value = {key: value}
        else:
            self._value = value

    @property
    def value(self):
        return self._value


class MeterStandardDeviationTest(unittest.TestCase):
    def test_exception(self):
        meter = ExceptionMeter()
        trigger = MeterStandardDeviation(meter, 0.1, 25)
        self.assertEqual(False, trigger())

    def test_keyed(self):
        meter = MockMeter()
        trigger = MeterStandardDeviation(meter, 0.1, 3, meter_key='key')
        meter.set_value(0, key='key')
        self.assertEqual(False, trigger())
        self.assertEqual(False, trigger())
        self.assertEqual(True, trigger())

    def test_not_enough_values(self):
        meter = MockMeter()
        trigger = MeterStandardDeviation(meter, 0.1, 25)
        meter.set_value(0)
        self.assertEqual(False, trigger())
        self.assertEqual(False, trigger())
        self.assertEqual(False, trigger())

    def test_above_threshold(self):
        meter = MockMeter()
        trigger = MeterStandardDeviation(meter, 0.01, 3)

        meter.set_value(1)
        self.assertEqual(False, trigger())
        meter.set_value(2)
        self.assertEqual(False, trigger())
        meter.set_value(3)
        self.assertEqual(False, trigger())

    def test_below_threshold(self):
        meter = MockMeter()
        trigger = MeterStandardDeviation(meter, 1, 3)

        meter.set_value(1)
        self.assertEqual(False, trigger())
        meter.set_value(2)
        self.assertEqual(False, trigger())
        meter.set_value(3)
        self.assertEqual(True, trigger())
        meter.set_value(3)
        self.assertEqual(True, trigger())


if __name__ == '__main__':
    unittest.main()
