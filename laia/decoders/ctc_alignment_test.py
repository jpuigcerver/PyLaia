from __future__ import absolute_import

import math
import unittest

import numpy as np
from laia.decoders.ctc_alignment import ctc_alignment


class CTCAlignmentTest(unittest.TestCase):
    def setUp(self):
        self._logp_matrix = np.log(
            np.array(
                [[0.3, 0.5, 0.2], [0.4, 0.5, 0.1], [0.5, 0.1, 0.4], [0.1, 0.7, 0.2]]
            )
        )

    def test_empty_reference(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.3 * 0.4 * 0.5 * 0.1))
        self.assertEqual([0, 0, 0, 0], best_alig)

    def test_empty_reference2(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [], ctc_sym=2)
        np.testing.assert_almost_equal(best_logp, math.log(0.2 * 0.1 * 0.4 * 0.2))
        self.assertEqual([2, 2, 2, 2], best_alig)

    def test_single_label(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [1], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.3 * 0.4 * 0.5 * 0.7))
        self.assertEqual([0, 0, 0, 1], best_alig)

    def test_single_label2(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [2], ctc_sym=1)
        np.testing.assert_almost_equal(best_logp, math.log(0.5 * 0.5 * 0.4 * 0.7))
        self.assertEqual([1, 1, 2, 1], best_alig)

    def test_repeated_label(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [2, 2], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.2 * 0.4 * 0.5 * 0.2))
        self.assertEqual([2, 0, 0, 2], best_alig)

    def test_regular(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [1, 2, 2], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.5 * 0.1 * 0.5 * 0.2))
        self.assertEqual([1, 2, 0, 2], best_alig)

    def test_regular2(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [1, 0], ctc_sym=2)
        np.testing.assert_almost_equal(best_logp, math.log(0.5 * 0.5 * 0.5 * 0.2))
        self.assertEqual([1, 1, 0, 2], best_alig)

    def test_regular3(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [0, 2, 0, 2], ctc_sym=1)
        np.testing.assert_almost_equal(best_logp, math.log(0.3 * 0.1 * 0.5 * 0.2))
        self.assertEqual([0, 2, 0, 2], best_alig)


if __name__ == "__main__":
    unittest.main()
