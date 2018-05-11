import unittest

import numpy as np
import torch
from laia.utils.phoc import unigram_phoc, probabilistic_phoc_relevance


class UnigramPHOCTest(unittest.TestCase):
    def test_level_1(self):
        d = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}

        phoc = unigram_phoc('12345', unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (1, 1, 1, 1, 1))

        phoc = unigram_phoc('34', unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (0, 0, 1, 1, 0))

        phoc = unigram_phoc('1234512345', unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (1, 1, 1, 1, 1))

    def test_level_2(self):
        d = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}

        phoc = unigram_phoc('12345', unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 1, 1, 0, 0, 0, 0, 1, 1, 1))

        phoc = unigram_phoc('34', unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (0, 0, 1, 0, 0, 0, 0, 0, 1, 0))

        phoc = unigram_phoc('1234512345', unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

    def test_missing_unigram_exception(self):
        d = {'1': 0, '2': 1, '4': 2, '5': 3}
        with self.assertRaises(KeyError):
            unigram_phoc('12345', unigram_map=d, unigram_levels=[1])

    def test_missing_unigram_warning(self):
        d = {'1': 0, '2': 1, '4': 2, '5': 3}
        phoc = unigram_phoc('12345', unigram_map=d, unigram_levels=[1],
                            ignore_missing=True)
        self.assertEqual(phoc, (1, 1, 1, 1))


class ProbabilisticPHOCTest(unittest.TestCase):
    def test_probabilistic_phoc_relevance(self):
        a = torch.DoubleTensor([0.5, 0.4, 0.4]).log_()
        b = torch.DoubleTensor([0.3, 0.7, 0.9]).log_()
        result = probabilistic_phoc_relevance(a, b)
        expected = np.log(
            (.5 * .3 + .5 * .7) * (.4 * .7 + .6 * .3) * (.4 * .9 + .6 * .1))
        np.testing.assert_almost_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
