import unittest

from laia.callbacks.meters.sequence_error import SequenceError, char_to_word_seq


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


class SequenceErrorTest(unittest.TestCase):
    def test_single_string(self):
        err = SequenceError()
        ref = ["home"]
        hyp = ["house"]
        err.add(ref, hyp)
        self.assertEqual(err.value, 0.5)

    def test_single_list(self):
        err = SequenceError()
        ref = [[1, 2, 3, 4]]
        hyp = [[1, 2, 5, 6, 4]]
        err.add(ref, hyp)
        self.assertEqual(err.value, 0.5)

    def test_empty_hyp_list(self):
        err = SequenceError()
        ref = [[[1, 2, 3, 4], [4, 3, 2, 1]]]
        hyp = [[]]
        err.add(ref, hyp)
        self.assertEqual(err.value, 1)

    def test_empty_hyp_string(self):
        err = SequenceError()
        ref = [["test", "thing"]]
        hyp = [[]]
        err.add(ref, hyp)
        self.assertEqual(err.value, 1)

    def test_multiple(self):
        err = SequenceError()
        ref = [["the", "house", "is", "blue"], ["my", "dog", "is", "black"]]
        hyp = [["the", "home", "is", "white"], ["my", "dog", "is", "not", "black"]]
        err.add(ref, hyp)
        self.assertEqual(err.value, (2 + 1) / (4 + 4))

    def test_multiple_calls(self):
        err = SequenceError()
        ref = [["the", "house", "is", "blue"], ["my", "dog", "is", "black"]]
        hyp = [["the", "home", "is", "white"], ["my", "dog", "is", "not", "black"]]
        err.add(ref, hyp)
        err.add(["home"], ["house"])
        self.assertEqual(err.value, (2 + 1 + 2) / (4 + 4 + 4))


if __name__ == "__main__":
    unittest.main()
