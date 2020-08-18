import unittest

import torch

from laia.decoders import CTCGreedyDecoder


class CTCGreedyDecoderTest(unittest.TestCase):
    def test(self):
        x = torch.tensor(
            [
                [[1.0, 3.0, -1.0, 0.0]],
                [[-1.0, 2.0, -2.0, 3.0]],
                [[1.0, 5.0, 9.0, 2.0]],
                [[-1.0, -2.0, -3.0, -4.0]],
            ]
        )
        decoder = CTCGreedyDecoder()
        r = decoder(x)
        e = [[1, 3, 2]]
        self.assertEqual(e, r)

    def test_segmentation_empty(self):
        s = CTCGreedyDecoder.compute_segmentation([])
        e = []
        self.assertEqual(e, s)

    def test_segmentation_one_symbol(self):
        x = [0]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 1]
        self.assertEqual(e, s)
        x = [1]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 1]
        self.assertEqual(e, s)

    def test_segmentation_only_zeros(self):
        x = [0, 0, 0]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 3]
        self.assertEqual(e, s)

    def test_segmentation_only_symbols(self):
        x = [1, 1, 1]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 3]
        self.assertEqual(e, s)
        x = [1, 2, 3]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 1, 2, 3]
        self.assertEqual(e, s)

    def test_segmentation(self):
        x = [1, 1, 0, 0, 0, 2, 0, 0, 3]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 2, 6, 9]
        self.assertEqual(e, s)
        x = [1, 2, 0, 0, 3, 2, 0, 0, 3]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 1, 2, 5, 6, 9]
        self.assertEqual(e, s)
        x = [0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 4, 8, 11, 13]
        self.assertEqual(e, s)
        x = [0, 0, 0, 2, 2]
        s = CTCGreedyDecoder.compute_segmentation(x)
        e = [0, 5]
        self.assertEqual(e, s)


if __name__ == "__main__":
    unittest.main()
