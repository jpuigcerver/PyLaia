from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import reduce

import torch
from torch._six import string_classes, int_classes
from torch.utils.data.dataloader import numpy_type_map

PaddedTensor = collections.namedtuple('PaddedTensor', ['data', 'sizes'])


def _get_max_size_and_check_batch_tensor(batch, expected_shape):
    # All tensors in the batch must have the same number of dimensions
    assert all(map(lambda x: x.dim() == batch[0].dim(), batch))
    max_sizes = [len(batch)]
    for d in range(batch[0].dim()):
        maxv, minv = reduce(
            lambda m, x: (
                m[0] if m[0] >= x.size()[d] else x.size()[d],
                m[1] if m[1] <= x.size()[d] else x.size()[d]),
            batch,
            (0, float('inf')))
        if expected_shape is None or expected_shape[d] is None:
            max_sizes.append(maxv)
        else:
            assert maxv == expected_shape[d] and minv == expected_shape[d]
            max_sizes.append(expected_shape[d])
    return max_sizes


class PaddingCollater(object):
    def __init__(self, padded_shapes, sort_key=None):
        self._padded_shapes = padded_shapes
        self._sort_key = sort_key

    def __call__(self, batch):
        if self._sort_key:
            batch = sorted(batch, key=self._sort_key)
        return self._collate(batch, self._padded_shapes)

    def _collate(self, batch, padded_shapes):
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if torch.is_tensor(batch[0]):
            max_sizes = _get_max_size_and_check_batch_tensor(batch,
                                                             padded_shapes)
            out = batch[0].new(*max_sizes).zero_()
            for i, x in enumerate(batch):
                # Change this to handle arbitrary dimensions.
                if x.dim() == 1:
                    out[i][:x.size(0)] = x
                elif x.dim() == 2:
                    out[i][:x.size(0), :x.size(1)] = x
                elif x.dim() == 3:
                    out[i][:x.size(0), :x.size(1), :x.size(2)] = x
                else:
                    raise NotImplementedError('This is not implemented')
            return PaddedTensor(data=out,
                                sizes=torch.stack([torch.LongTensor(list(x.size())) for x in batch]))
        elif (elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_'
              and elem_type.__name__ != 'string_'):
            elem = batch[0]
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](
                    list(map(py_type, batch)))

            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return self._collate([torch.from_numpy(b) for b in batch],
                                     padded_shapes)
        elif isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], collections.Mapping):
            out = {}
            for key in batch[0]:
                if key in padded_shapes:
                    out[key] = self._collate([d[key] for d in batch],
                                             padded_shapes[key])
                else:
                    out[key] = [d[key] for d in batch]
            return out
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [self._collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))
