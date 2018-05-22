from __future__ import absolute_import
from builtins import range

import math
import unittest

from numpy import logaddexp
from numpy.testing import assert_almost_equal

from laia.utils.discrete_normal_distribution import DiscreteNormalDistribution


class DiscreteNormalDistributionTest(unittest.TestCase):

    def testConstructor(self):
        d = DiscreteNormalDistribution(25, 1)
        self.assertEqual(d.mean, 25)
        self.assertEqual(d.var, 1)

    def testPdf(self):
        d = DiscreteNormalDistribution(8, 4, eps=1e-12)
        assert_almost_equal(d.log_pdf(8), -1.6120768644453298)
        assert_almost_equal(d.log_pdf(0.0), -9.612076921466052)
        assert_almost_equal(d.log_pdf(-1.0), -float("inf"))

        assert_almost_equal(d.pdf(8), math.exp(-1.6120768644453298))
        assert_almost_equal(d.pdf(0.0), math.exp(-9.612076921466052))
        assert_almost_equal(d.pdf(-1.0), 0.0)

    def testSum(self):
        d = DiscreteNormalDistribution(20, 2, eps=1e-12)
        acc = -float("inf")
        for x in range(1000):
            acc = logaddexp(acc, d.log_pdf(x))
        # The sum should be ~1.0 (log = 0.0)
        assert_almost_equal(acc, 0.0)


if __name__ == "__main__":
    unittest.main()
