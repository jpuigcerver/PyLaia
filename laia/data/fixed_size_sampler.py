from __future__ import absolute_import

import torch
from torch.utils.data.sampler import Sampler


class FixedSizeSampler(Sampler):
    """Randomly resample elements from a data source up to a fixed number of
    samples.

    In each iterator, the samples are randomly sorted. If `num_samples` is
    greater than the number of samples in the `data_source`, there will be
    some repetitions in the new dataset. Otherwise, the samples will be unique.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to resample from the dataset
    """
    def __init__(self, data_source, num_samples):
        super(FixedSizeSampler, self).__init__(data_source)
        self._data_source = data_source
        self._num_samples = num_samples

    def __iter__(self):
        idxs = []
        while len(idxs) < self._num_samples:
            idxs.extend(torch.randperm(len(self._data_source)).tolist())
        return iter(idxs[:self._num_samples])

    def __len__(self):
        return self._num_samples
