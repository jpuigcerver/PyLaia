import unittest

from laia.utils.phoc import unigram_phoc, new_unigram_phoc


class UnigramPHOCTest(unittest.TestCase):

    def test_level_1(self):
        d = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        phoc = unigram_phoc("12345", unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (1, 1, 1, 1, 1))

        phoc = unigram_phoc("34", unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (0, 0, 1, 1, 0))

        phoc = unigram_phoc("1234512345", unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (1, 1, 1, 1, 1))

    def test_level_2(self):
        d = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        phoc = unigram_phoc("12345", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 1, 1, 0, 0, 0, 0, 1, 1, 1))

        phoc = unigram_phoc("34", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (0, 0, 1, 0, 0, 0, 0, 0, 1, 0))

        phoc = unigram_phoc("1234512345", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

    def test_missing_unigram_exception(self):
        d = {"1": 0, "2": 1, "4": 2, "5": 3}
        with self.assertRaises(KeyError):
            unigram_phoc("12345", unigram_map=d, unigram_levels=[1])

    def test_missing_unigram_warning(self):
        d = {"1": 0, "2": 1, "4": 2, "5": 3}
        phoc = unigram_phoc(
            "12345", unigram_map=d, unigram_levels=[1], ignore_missing=True
        )
        self.assertEqual(phoc, (1, 1, 1, 1))


class NewUnigramPHOCTest(unittest.TestCase):

    def test_level_1(self):
        d = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        phoc = new_unigram_phoc("12345", unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (1, 1, 1, 1, 1))

        phoc = new_unigram_phoc("34", unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (0, 0, 1, 1, 0))

        phoc = new_unigram_phoc("1234512345", unigram_map=d, unigram_levels=[1])
        self.assertEqual(phoc, (1, 1, 1, 1, 1))

    def test_level_2(self):
        d = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        phoc = new_unigram_phoc("12345", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 1, 1, 0, 0, 0, 0, 0, 1, 1))

        phoc = new_unigram_phoc("34", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (0, 0, 1, 0, 0, 0, 0, 0, 1, 0))

        phoc = new_unigram_phoc("1234512345", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

        phoc = new_unigram_phoc("1", unigram_map=d, unigram_levels=[2])
        self.assertEqual(phoc, (1, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def test_missing_unigram_exception(self):
        d = {"1": 0, "2": 1, "4": 2, "5": 3}
        with self.assertRaises(KeyError):
            new_unigram_phoc("12345", unigram_map=d, unigram_levels=[1])

    def test_missing_unigram_warning(self):
        d = {"1": 0, "2": 1, "4": 2, "5": 3}
        phoc = new_unigram_phoc(
            "12345", unigram_map=d, unigram_levels=[1], ignore_missing=True
        )
        self.assertEqual(phoc, (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
