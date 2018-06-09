from __future__ import absolute_import

import math
import unittest

import numpy as np
from laia.decoders.ctc_alignment import ctc_alignment


class CTCAlignmentTest(unittest.TestCase):
    def setUp(self):
        self._logp_matrix = np.log(
            np.asarray(
                [[0.3, 0.5, 0.2], [0.4, 0.5, 0.1], [0.5, 0.1, 0.4], [0.1, 0.7, 0.2]]
            )
        )

    def empty_reference_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.3 * 0.4 * 0.5 * 0.1))
        self.assertEqual([0, 0, 0, 0], best_alig)

    def empty_reference2_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [], ctc_sym=2)
        np.testing.assert_almost_equal(best_logp, math.log(0.2 * 0.1 * 0.4 * 0.2))
        self.assertEqual([2, 2, 2, 2], best_alig)

    def single_label_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [1], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.3 * 0.4 * 0.5 * 0.7))
        self.assertEqual([0, 0, 0, 1], best_alig)

    def single_label2_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [2], ctc_sym=1)
        np.testing.assert_almost_equal(best_logp, math.log(0.5 * 0.5 * 0.4 * 0.7))
        self.assertEqual([1, 1, 2, 1], best_alig)

    def repeated_label_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [2, 2], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.2 * 0.4 * 0.5 * 0.2))
        self.assertEqual([2, 0, 0, 2], best_alig)

    def regular_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [1, 2, 2], ctc_sym=0)
        np.testing.assert_almost_equal(best_logp, math.log(0.5 * 0.1 * 0.5 * 0.2))
        self.assertEqual([1, 2, 0, 2], best_alig)

    def regular2_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [1, 0], ctc_sym=2)
        np.testing.assert_almost_equal(best_logp, math.log(0.5 * 0.5 * 0.5 * 0.2))
        self.assertEqual([1, 1, 0, 2], best_alig)

    def regular4_test(self):
        best_logp, best_alig = ctc_alignment(self._logp_matrix, [0, 2, 0, 2], ctc_sym=1)
        np.testing.assert_almost_equal(best_logp, math.log(0.3 * 0.1 * 0.5 * 0.2))
        self.assertEqual([0, 2, 0, 2], best_alig)


if __name__ == "__main__":
    unittest.main()
