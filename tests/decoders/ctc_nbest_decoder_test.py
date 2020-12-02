import unittest

import torch

from laia.decoders import CTCNBestDecoder


class CTCNBestDecoderTest(unittest.TestCase):
    def test(self):
        x = torch.tensor(
            [
                [[1.0, 3.0, -1.0, 0.0]],
                [[-1.0, 2.0, -2.0, 3.0]],
                [[1.0, 5.0, 9.0, 2.0]],
                [[-1.0, -2.0, -3.0, -4.0]],
            ]
        )
        decoder = CTCNBestDecoder(4)
        r = decoder(x)
        paths = ([1, 3, 2, 0], [1, 1, 2, 0], [1, 3, 2, 1], [1, 1, 2, 1])
        e = [[(sum(x[i, 0, v] for i, v in enumerate(p)).item(), p) for p in paths]]
        self.assertEqual(e, r)


if __name__ == "__main__":
    unittest.main()
