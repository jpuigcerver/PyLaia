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
        self.assertEqual(e, r["hyp"])

    def test_prob(self):
        x = torch.tensor([[[0.3, 0.6, 0.1]], [[0.6, 0.3, 0.2]]]).log()
        decoder = CTCGreedyDecoder()
        r = decoder(x, segmentation=True, apply_softmax=False)
        e = [[1]]
        self.assertEqual(e, r["hyp"])
        # Check actual loss prob
        paths = torch.tensor(
            [x[0, 0, a] + x[1, 0, b] for a, b in ((0, 1), (1, 0), (1, 1))]
        )
        loss = torch.nn.functional.ctc_loss(
            x, torch.tensor(e), torch.tensor([2]), torch.tensor([1]), reduction="none"
        )
        loss_prob = loss.neg().exp().item()
        path_prob = paths.exp().sum().item()

        torch.testing.assert_allclose(loss_prob, path_prob)
        # Check 1best prob against loss with input_length = 1
        loss = torch.nn.functional.ctc_loss(
            x, torch.tensor(e), torch.tensor([1]), torch.tensor([1]), reduction="none"
        )
        loss_prob = loss.neg().exp().item()
        torch.testing.assert_allclose(
            loss_prob, [p.mean() for p in r["prob-segmentation"]][0].item()
        )

    def test_batch(self):
        x = torch.tensor([[[0.3, 0.6], [0.5, 0.9]], [[0.6, 0.3], [0.6, 0.9]]]).log()
        decoder = CTCGreedyDecoder()
        r = decoder(x, segmentation=True, apply_softmax=False)
        e = [[1], [1]]
        self.assertEqual(e, r["hyp"])
        # note: checking with ctc_loss does not work for every x
        loss = torch.nn.functional.ctc_loss(
            x,
            torch.tensor(e),
            torch.tensor([1, 1]),
            torch.tensor([1, 1]),
            reduction="none",
        )
        e = loss.neg().exp()
        r = torch.tensor([p.mean() for p in r["prob-segmentation"]])
        torch.testing.assert_allclose(r, e)

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
