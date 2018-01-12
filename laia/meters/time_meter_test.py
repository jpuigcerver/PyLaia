import time
import unittest

from .time_meter import TimeMeter

class TimeMeterTest(unittest.TestCase):
    def testMeter(self):
        m = TimeMeter()
        time.sleep(1)
        t = m.value
        # Check that the timer has measured ~1 second.
        self.assertGreaterEqual(t, 1.0)
        self.assertLess(t, 1.1)

if __name__ == '__main__':
    unittest.main()
