from __future__ import absolute_import

from laia.data.fixed_size_sampler import FixedSizeSampler

import unittest


class DummyDataset(object):
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size


class FixedSizeSamplerTest(unittest.TestCase):
    def test_simple(self):
        sampler = FixedSizeSampler(DummyDataset(2), 2)
        self.assertEqual(2, len(sampler))
        it = iter(sampler)
        self.assertEqual({0, 1}, set(x for x in it))

    def test_fewer_elements_than_source(self):
        sampler = FixedSizeSampler(DummyDataset(10), 2)
        self.assertEqual(2, len(sampler))
        it = iter(sampler)
        self.assertEqual(2, len([x for x in it]))

    def test_more_elements_than_source(self):
        sampler = FixedSizeSampler(DummyDataset(3), 10)
        self.assertEqual(10, len(sampler))
        it = iter(sampler)
        samples = [x for x in it]
        self.assertEqual(10, len(samples))
        self.assertEqual({0, 1, 2}, set(samples))


if __name__ == "__main__":
    unittest.main()
