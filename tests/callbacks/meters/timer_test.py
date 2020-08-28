import time
import unittest

from laia.callbacks.meters.timer import Timer


class TimerTest(unittest.TestCase):
    def test(self):
        m = Timer()
        time.sleep(1)
        t = m.value
        # Check that the timer has measured ~1 second.
        self.assertGreaterEqual(t, 1.0)
        self.assertLess(t, 1.1)


if __name__ == "__main__":
    unittest.main()
