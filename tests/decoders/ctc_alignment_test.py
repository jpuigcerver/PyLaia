import math
import unittest

import torch

from laia.decoders import ctc_alignment


class CTCAlignmentTest(unittest.TestCase):
    def setUp(self):
        self._logp = torch.tensor(
            [[0.3, 0.5, 0.2], [0.4, 0.5, 0.1], [0.5, 0.1, 0.4], [0.1, 0.7, 0.2]]
        ).log_()

    def test_empty_reference(self):
        best_logp, best_alig = ctc_alignment(self._logp, [], ctc_sym=0)
        torch.testing.assert_allclose(best_logp, math.log(0.3 * 0.4 * 0.5 * 0.1))
        self.assertEqual([0, 0, 0, 0], best_alig)

    def test_empty_reference2(self):
        best_logp, best_alig = ctc_alignment(self._logp, [], ctc_sym=2)
        torch.testing.assert_allclose(best_logp, math.log(0.2 * 0.1 * 0.4 * 0.2))
        self.assertEqual([2, 2, 2, 2], best_alig)

    def test_single_label(self):
        best_logp, best_alig = ctc_alignment(self._logp, [1], ctc_sym=0)
        torch.testing.assert_allclose(best_logp, math.log(0.3 * 0.4 * 0.5 * 0.7))
        self.assertEqual([0, 0, 0, 1], best_alig)

    def test_single_label2(self):
        best_logp, best_alig = ctc_alignment(self._logp, [2], ctc_sym=1)
        torch.testing.assert_allclose(best_logp, math.log(0.5 * 0.5 * 0.4 * 0.7))
        self.assertEqual([1, 1, 2, 1], best_alig)

    def test_repeated_label(self):
        best_logp, best_alig = ctc_alignment(self._logp, [2, 2], ctc_sym=0)
        torch.testing.assert_allclose(best_logp, math.log(0.2 * 0.4 * 0.5 * 0.2))
        self.assertEqual([2, 0, 0, 2], best_alig)

    def test_regular(self):
        best_logp, best_alig = ctc_alignment(self._logp, [1, 2, 2], ctc_sym=0)
        torch.testing.assert_allclose(best_logp, math.log(0.5 * 0.1 * 0.5 * 0.2))
        self.assertEqual([1, 2, 0, 2], best_alig)

    def test_regular2(self):
        best_logp, best_alig = ctc_alignment(self._logp, [1, 0], ctc_sym=2)
        torch.testing.assert_allclose(best_logp, math.log(0.5 * 0.5 * 0.5 * 0.2))
        self.assertEqual([1, 1, 0, 2], best_alig)

    def test_regular3(self):
        best_logp, best_alig = ctc_alignment(self._logp, [0, 2, 0, 2], ctc_sym=1)
        torch.testing.assert_allclose(best_logp, math.log(0.3 * 0.1 * 0.5 * 0.2))
        self.assertEqual([0, 2, 0, 2], best_alig)


if __name__ == "__main__":
    unittest.main()
