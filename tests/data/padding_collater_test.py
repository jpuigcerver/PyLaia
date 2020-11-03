import unittest

import numpy as np
import pytest
import torch

from laia.data import PaddedTensor, PaddingCollater


@pytest.mark.parametrize(
    ["data", "sizes", "match"],
    [
        (None, None, None),
        (torch.empty(1), None, None),
        (torch.empty(1), torch.tensor(1), r"PaddedTensor.sizes must have 2 dimensions"),
        (
            torch.empty(1),
            torch.empty(1, 1),
            r"PaddedTensor.sizes is incorrect: .* found=1",
        ),
        (
            torch.empty(1),
            torch.empty(1, 4),
            r"PaddedTensor.sizes is incorrect: .* found=4",
        ),
        (torch.empty(1), torch.empty(2, 2), r"Batch size 2 .* batch 1"),
    ],
)
def test_padded_tensor_assertions(data, sizes, match):
    with pytest.raises(AssertionError, match=match):
        PaddedTensor.build(data, sizes)


def test_padded_tensor_repr():
    sizes = torch.randint(3, size=(2, 3))
    t = PaddedTensor.build(torch.empty(2, 1, 5, 5), sizes)
    assert (
        repr(t)
        == f"PaddedTensor(data.size()=[2, 1, 5, 5], sizes={sizes.tolist()}, device=cpu)"
    )


class TestPaddingCollater(unittest.TestCase):
    def test_max_sizes_all_sizes_set(self):
        C = 1
        batch = [torch.rand(C, 20, 40), torch.rand(C, 20, 40), torch.rand(C, 20, 40)]
        sizes = (C, 20, 40)
        max_sizes = PaddingCollater.get_max_sizes(batch, sizes=sizes)
        expected = (len(batch), C, 20, 40)
        self.assertEqual(expected, max_sizes)

    def test_max_sizes_incorrect_sizes(self):
        batch = [torch.rand(1, 20, 40), torch.rand(1, 20, 45), torch.rand(1, 20, 40)]
        sizes = (1, 20, 40)
        with self.assertRaises(AssertionError):
            PaddingCollater.get_max_sizes(batch, sizes=sizes)

    def test_max_sizes_incorrect_dim(self):
        batch = [torch.rand(1, 20, 40), torch.rand(1, 1, 20, 45)]
        with self.assertRaises(AssertionError):
            PaddingCollater.get_max_sizes(batch)

    def test_max_sizes_with_channels_size(self):
        C = 1
        batch = [torch.rand(C, 20, 40), torch.rand(C, 25, 30), torch.rand(C, 15, 35)]
        sizes = (C, None, None)
        max_sizes = PaddingCollater.get_max_sizes(batch, sizes=sizes)
        expected = (len(batch), C, 25, 40)
        self.assertEqual(expected, max_sizes)

    def test_max_sizes_no_sizes(self):
        batch = [torch.rand(3, 20, 40), torch.rand(3, 25, 30), torch.rand(5, 15, 35)]
        max_sizes = PaddingCollater.get_max_sizes(batch)
        expected = (len(batch), 5, 25, 40)
        self.assertEqual(expected, max_sizes)

    def check_collated(self, batch, max_sizes, collated):
        self.assertEqual(collated.size(), max_sizes)
        for i, x in enumerate(batch):
            torch.testing.assert_allclose(
                collated[i, : x.size(0), : x.size(1), : x.size(2)], x
            )
            torch.testing.assert_allclose(
                sum(collated[i, x.size(0) :, x.size(1) :, x.size(2) :]), 0
            )

    def test_collate_tensors(self):
        batch = [torch.rand(3, 20, 40), torch.rand(3, 25, 30), torch.rand(5, 15, 35)]
        max_sizes = (len(batch), 5, 25, 40)
        collated = PaddingCollater.collate_tensors(batch, max_sizes)
        self.check_collated(batch, max_sizes, collated)

    def test_collate_with_tensor(self):
        sizes = (None, None, None)
        collate_fn = PaddingCollater(sizes)
        batch = [torch.rand(3, 20, 40), torch.rand(3, 25, 30), torch.rand(5, 15, 35)]
        x, xs = collate_fn(batch)
        for i in range(len(batch)):
            self.assertEqual(list(batch[i].size()), xs[i].tolist())
        self.check_collated(batch, (3, 5, 25, 40), x)

    def test_collate_with_tensor_and_fixed_sizes(self):
        sizes = (1, 20, 40)
        collate_fn = PaddingCollater(sizes)
        batch = [torch.rand(1, 20, 40), torch.rand(1, 20, 40), torch.rand(1, 20, 40)]
        x = collate_fn(batch)
        torch.testing.assert_allclose(x, torch.stack(batch))

    def test_collate_with_list(self):
        sizes = [(None, None, None), (1, None, None)]
        collate_fn = PaddingCollater(sizes)
        batch = [
            [torch.rand(3, 20, 40), torch.rand(3, 25, 30), torch.rand(5, 15, 35)],
            [torch.rand(1, 20, 40), torch.rand(1, 25, 30), torch.rand(1, 15, 35)],
        ]
        expected = [(3, 5, 25, 40), (3, 1, 25, 40)]
        for i, (x, xs) in enumerate(collate_fn(batch)):
            for j in range(len(batch)):
                self.assertEqual(list(batch[i][j].size()), xs[j].tolist())
            self.check_collated(batch[i], expected[i], x)

    def test_collate_with_numpy(self):
        sizes = (1, 20, 40)
        collate_fn = PaddingCollater(sizes)
        batch = [
            np.random.rand(1, 20, 40),
            np.random.rand(1, 20, 40),
            np.random.rand(1, 20, 40),
        ]
        x = collate_fn(batch)
        torch.testing.assert_allclose(
            x, torch.stack([torch.from_numpy(x) for x in batch])
        )

    def test_collate_with_dict(self):
        sizes = {"img": (3, None, None)}
        collate_fn = PaddingCollater(sizes)
        batch = [
            {"img": torch.rand(3, 20, 40)},
            {"img": torch.rand(3, 25, 30)},
            {"img": torch.rand(3, 15, 35)},
        ]
        x, xs = collate_fn(batch)["img"]
        for i, b in enumerate(batch):
            self.assertEqual(list(b["img"].size()), xs[i].tolist())
        self.check_collated([b["img"] for b in batch], (3, 3, 25, 40), x)


if __name__ == "__main__":
    unittest.main()
