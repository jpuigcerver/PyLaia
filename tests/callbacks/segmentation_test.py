import unittest

from laia.callbacks.segmentation import char_segmentation, word_segmentation


class CharSegmentationTest(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(AssertionError):
            char_segmentation([""], [], 1)

    def test(self):
        txt = ["a", "b", "c"]
        seg = [0, 3, 5, 7, 10]
        x = char_segmentation(txt, seg, 1)
        e = [("a", 1, 1, 2, 1), ("b", 3, 1, 4, 1), ("c", 5, 1, 6, 1)]
        self.assertEqual(e, x)

    def test_scaling(self):
        txt = ["a", "b", "c"]
        seg = [0, 3, 5, 7, 10]
        x = char_segmentation(txt, seg, 1, width=100)
        e = [("a", 1, 1, 29, 1), ("b", 30, 1, 49, 1), ("c", 50, 1, 69, 1)]
        self.assertEqual(e, x)

    def test_scaling_error(self):
        with self.assertRaises(AssertionError):
            char_segmentation(["a"], [0, 1, 100], 1, width=50)


class WordSegmentationTest(unittest.TestCase):
    def test_empty(self):
        x = word_segmentation([], " ", include_spaces=True)
        self.assertEqual(x, [])

    def test_different_y1(self):
        s = [("a", 1, 1, 3, 10), ("b", 4, 5, 10, 10)]
        with self.assertRaises(AssertionError):
            word_segmentation(s, " ")

    def test_different_y2(self):
        s = [("a", 1, 1, 3, 10), ("b", 4, 1, 10, 5)]
        with self.assertRaises(AssertionError):
            word_segmentation(s, " ")

    def test_x_not_contiguous(self):
        s = [("a", 1, 1, 3, 10), ("b", 3, 1, 10, 10)]
        with self.assertRaises(AssertionError):
            word_segmentation(s, " ")
        s = [("a", 1, 1, 3, 10), ("b", 5, 1, 10, 10)]
        with self.assertRaises(AssertionError):
            word_segmentation(s, " ")

    def test_one_element(self):
        x = word_segmentation([("a", 1, 1, 10, 10)], " ", include_spaces=True)
        self.assertEqual(x, [("a", 1, 1, 10, 10)])

    def test_with_spaces(self):
        s = [
            ("a", 1, 1, 2, 10),
            ("b", 3, 1, 3, 10),
            (" ", 4, 1, 5, 10),
            ("c", 6, 1, 800, 10),
        ]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [("ab", 1, 1, 3, 10), (" ", 4, 1, 5, 10), ("c", 6, 1, 800, 10)]
        self.assertEqual(e, x)

    def test_without_spaces(self):
        s = [
            ("a", 1, 1, 2, 10),
            ("b", 3, 1, 3, 10),
            (" ", 4, 1, 5, 10),
            ("c", 6, 1, 800, 10),
        ]
        x = word_segmentation(s, " ", include_spaces=False)
        e = [("ab", 1, 1, 3, 10), ("c", 6, 1, 800, 10)]
        self.assertEqual(e, x)

    def test_space_at_beginning(self):
        s = [(" ", 1, 1, 2, 1), ("b", 3, 1, 3, 1), ("c", 4, 1, 5, 1)]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [(" ", 1, 1, 2, 1), ("bc", 3, 1, 5, 1)]
        self.assertEqual(e, x)
        x = word_segmentation(s, " ", include_spaces=False)
        e = [("bc", 3, 1, 5, 1)]
        self.assertEqual(e, x)

    def test_space_at_end(self):
        s = [("a", 1, 1, 1, 1), ("b", 2, 1, 2, 1), (" ", 3, 1, 5, 1)]
        x = word_segmentation(s, " ", include_spaces=True)
        e = [("ab", 1, 1, 2, 1), (" ", 3, 1, 5, 1)]
        self.assertEqual(e, x)
        x = word_segmentation(s, " ", include_spaces=False)
        e = [("ab", 1, 1, 2, 1)]
        self.assertEqual(e, x)


if __name__ == "__main__":
    unittest.main()
