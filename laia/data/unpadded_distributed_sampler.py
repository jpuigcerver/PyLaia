# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# Originally seen in: https://github.com/pytorch/pytorch/issues/25162
# - Changed to subclass DistributedSampler.
# BEWARE of this sampler for the reasons described further down.
# The only current alternative AFAIK is to disable testing with more
# than one process, however, validation would still produce incorrect
# metric values. Hopefully this can be replaced soon when
# pytorch-lightning supports PyTorch's new join-based API. see:
# https://github.com/PyTorchLightning/pytorch-lightning/issues/3325

import torch
from torch.utils.data.distributed import DistributedSampler


class UnpaddedDistributedSampler(DistributedSampler):
    """
    A fork of the pytorch DistributedSampler that doesn't repeat data, instead
    allowing the number of batches per process to be off-by-one from each other.
    This makes this sampler usable for validation (it's deterministic and
    doesn't require shuffling). It is potentially unsafe to use this sampler for
    training, because during training the DistributedDataParallel syncs buffers
    on each forward pass, so it could freeze if one of the processes runs one
    fewer batch. During validation, buffers are only synced on the first batch,
    so this is safe to use as long as each process runs at least one batch. We
    verify this in an assert.

    Example::
        >>> sampler = UnpaddedDistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))
        self.total_size = len(self.dataset)
        # If any process has at least one batch, every other process needs to
        # have at least one batch, or the DistributedDataParallel could lock up.
        assert self.num_samples >= 1 or self.total_size == 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
