import unittest

from laia.utils.segmentation import word_segmentation


class SegmentationTest(unittest.TestCase):
    def test_empty(self):
        x = word_segmentation([], " ", include_spaces=True)
        self.assertEqual(x, [])

    def test_one_element(self):
        x = word_segmentation([("a", 0, 10)], " ", include_spaces=True)
        self.assertEqual(x, [("a", 0, 10)])

    def test_with_spaces(self):
        s = [("a", 0, 1), ("b", 1, 2), (" ", 2, 5), ("c", 5, 800)]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [("ab", 0, 2), (" ", 2, 5), ("c", 5, 800)]
        self.assertEqual(e, x)

    def test_without_spaces(self):
        s = [("a", 0, 1), ("b", 1, 2), (" ", 2, 5), ("c", 5, 800)]
        x = word_segmentation(s, " ", include_spaces=False)
        e = [("ab", 0, 2), ("c", 5, 800)]
        self.assertEqual(e, x)

    def test_space_at_beginning(self):
        s = [(" ", 0, 1), ("b", 1, 2), ("c", 2, 5)]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [(" ", 0, 1), ("bc", 1, 5)]
        self.assertEqual(e, x)
        x = word_segmentation(s, " ", include_spaces=False)
        e = [("bc", 1, 5)]
        self.assertEqual(e, x)

    def test_space_at_end(self):
        s = [("a", 0, 1), ("b", 1, 2), (" ", 2, 5)]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [("ab", 0, 2), (" ", 2, 5)]
        self.assertEqual(e, x)
        x = word_segmentation(s, " ", include_spaces=False)
        e = [("ab", 0, 2)]
        self.assertEqual(e, x)

    def test_bad_indices(self):
        s = [("a", 0, 1), ("b", 2, 3), (" ", 5, 5), ("c", 5, 800)]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [("ab", 0, 3), (" ", 5, 5), ("c", 5, 800)]
        self.assertEqual(e, x)


if __name__ == "__main__":
    unittest.main()
