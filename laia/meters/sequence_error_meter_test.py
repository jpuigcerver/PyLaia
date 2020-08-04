import unittest

from laia.meters import SequenceErrorMeter


class SequenceErrorMeterTest(unittest.TestCase):
    def test_single_string(self):
        err = SequenceErrorMeter()
        ref = ["home"]
        hyp = ["house"]
        err.add(ref, hyp)
        self.assertEqual(err.value, 0.5)

    def test_single_list(self):
        err = SequenceErrorMeter()
        ref = [[1, 2, 3, 4]]
        hyp = [[1, 2, 5, 6, 4]]
        err.add(ref, hyp)
        self.assertEqual(err.value, 0.5)

    def test_empty_hyp_list(self):
        err = SequenceErrorMeter()
        ref = [[[1, 2, 3, 4], [4, 3, 2, 1]]]
        hyp = [[]]
        err.add(ref, hyp)
        self.assertEqual(err.value, 1)

    def test_empty_hyp_string(self):
        err = SequenceErrorMeter()
        ref = [["test", "thing"]]
        hyp = [[]]
        err.add(ref, hyp)
        self.assertEqual(err.value, 1)

    def test_multiple(self):
        err = SequenceErrorMeter()
        ref = [["the", "house", "is", "blue"], ["my", "dog", "is", "black"]]
        hyp = [["the", "home", "is", "white"], ["my", "dog", "is", "not", "black"]]
        err.add(ref, hyp)
        self.assertEqual(err.value, (2 + 1) / (4 + 4))

    def test_multiple_calls(self):
        err = SequenceErrorMeter()
        ref = [["the", "house", "is", "blue"], ["my", "dog", "is", "black"]]
        hyp = [["the", "home", "is", "white"], ["my", "dog", "is", "not", "black"]]
        err.add(ref, hyp)
        err.add(["home"], ["house"])
        self.assertEqual(err.value, (2 + 1 + 2) / (4 + 4 + 4))

    def test_state_dict(self):
        err = SequenceErrorMeter()
        ref = [[1, 2, 3, 4]]
        hyp = [[1, 2, 5, 6, 4]]
        err.add(ref, hyp)
        err2 = SequenceErrorMeter()
        err2.load_state_dict(err.state_dict())
        self.assertEqual(err.value, err2.value)


if __name__ == "__main__":
    unittest.main()
