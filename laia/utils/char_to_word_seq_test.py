import unittest

from laia.utils.char_to_word_seq import char_to_word_seq


class CharToWordSeqTest(unittest.TestCase):
    def test_empty_string(self):
        seq = ""
        delimiters = []
        actual = char_to_word_seq(seq, delimiters)
        expected = []
        self.assertEqual(actual, expected)

    def test_empty_list(self):
        seq = []
        delimiters = [" "]
        actual = char_to_word_seq(seq, delimiters)
        expected = []
        self.assertEqual(actual, expected)

    def test_empty_delimiters(self):
        seq = "hello my friend"
        delimiters = []
        actual = char_to_word_seq(seq, delimiters)
        expected = [
            ["h", "e", "l", "l", "o", " ", "m", "y", " ", "f", "r", "i", "e", "n", "d"]
        ]
        self.assertEqual(actual, expected)

    def test_with_text(self):
        seq = "hello my friend"
        delimiters = [" "]
        actual = char_to_word_seq(seq, delimiters)
        expected = [
            ["h", "e", "l", "l", "o"],
            ["m", "y"],
            ["f", "r", "i", "e", "n", "d"],
        ]
        self.assertEqual(actual, expected)

    def test_with_text_long_delimiter(self):
        seq = "hello<space>my friend"
        delimiters = ["<space>", " "]
        actual = char_to_word_seq(seq, delimiters)
        expected = [
            ["h", "e", "l", "l", "o"],
            ["m", "y"],
            ["f", "r", "i", "e", "n", "d"],
        ]
        self.assertEqual(actual, expected)

    def test_with_text_extra_whitespace(self):
        seq = " hello  my friend    "
        delimiters = [" "]
        actual = char_to_word_seq(seq, delimiters)
        expected = [
            ["h", "e", "l", "l", "o"],
            ["m", "y"],
            ["f", "r", "i", "e", "n", "d"],
        ]
        self.assertEqual(actual, expected)

    def test_with_numbers(self):
        seq = [-2, 1, 2, 3, -1, 1, 2, -2, 3]
        delimiters = [-1, -2]
        actual = char_to_word_seq(seq, delimiters)
        expected = [[1, 2, 3], [1, 2], [3]]
        self.assertEqual(actual, expected)

    def test_with_repeated_numbers(self):
        seq = [-2, 1, 2, 3, -1, -1, -1, 1, 2, -2, 3]
        delimiters = [-1, -2]
        actual = char_to_word_seq(seq, delimiters)
        expected = [[1, 2, 3], [1, 2], [3]]
        self.assertEqual(actual, expected)

    def test_with_numbers_extra_delimiter(self):
        seq = [-2, 1, 2, 3, -1, 1, 2, -2, 3]
        delimiters = [-1, -2, 80]
        actual = char_to_word_seq(seq, delimiters)
        expected = [[1, 2, 3], [1, 2], [3]]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
