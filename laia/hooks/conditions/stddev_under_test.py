from __future__ import absolute_import

import unittest

from laia.hooks.conditions import StdDevUnder
from laia.hooks.meters import Meter


class ExceptionMeter(Meter):

    @property
    def value(self):
        raise Exception


class MockMeter(Meter):

    def __init__(self):
        super(MockMeter, self).__init__()
        self._value = None

    def set_value(self, value, key=None):
        self._value = {key: value} if key else value

    @property
    def value(self):
        return self._value


class StdDevUnderTest(unittest.TestCase):

    def test_exception(self):
        meter = ExceptionMeter()
        cond = StdDevUnder(meter, 0.1, 25)

        self.assertEqual(False, cond())

    def test_with_key(self):
        meter = MockMeter()
        cond = StdDevUnder(meter, 0.1, 3, key="key")

        meter.set_value(0, key="key")
        self.assertEqual(False, cond())
        self.assertEqual(False, cond())
        self.assertEqual(True, cond())

    def test_not_enough_values(self):
        meter = MockMeter()
        trigger = StdDevUnder(meter, 0.1, 25)

        meter.set_value(0)
        self.assertEqual(False, trigger())
        self.assertEqual(False, trigger())
        self.assertEqual(False, trigger())

    def test_above_threshold(self):
        meter = MockMeter()
        cond = StdDevUnder(meter, 0.01, 3)

        meter.set_value(1)
        self.assertEqual(False, cond())
        meter.set_value(2)
        self.assertEqual(False, cond())
        meter.set_value(3)
        self.assertEqual(False, cond())

    def test_below_threshold(self):
        meter = MockMeter()
        cond = StdDevUnder(meter, 1, 3)

        meter.set_value(1)
        self.assertEqual(False, cond())
        meter.set_value(2)
        self.assertEqual(False, cond())
        meter.set_value(3)
        self.assertEqual(True, cond())
        meter.set_value(3)
        self.assertEqual(True, cond())


if __name__ == "__main__":
    unittest.main()
