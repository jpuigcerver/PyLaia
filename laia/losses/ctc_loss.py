from __future__ import absolute_import

import itertools
from typing import List, Union, Sequence

import torch
from torch.autograd import Variable, Function
import laia.common.logging as log
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from laia.losses.loss import Loss

try:
    from warpctc_pytorch import _CTC as _WARP_CTC
except ImportError:
    import warnings

    warnings.warn("Missing CTC loss function library")

FloatScalar = Union[float, torch.FloatTensor]

# TODO(jpuigcerver): Properly tests the logged messages.
_logger = log.get_logger(__name__)


def transform_output(output):
    # Size: T x N x D
    if isinstance(output, PackedSequence):
        acts, act_lens = pad_packed_sequence(output)
    elif torch.is_tensor(output) or isinstance(output, Variable):
        acts, act_lens = output, [output.size(0)] * output.size(1)
    else:
        raise NotImplementedError("Not implemented for type {}".format(type(output)))
    return acts, act_lens


def copy_valid_indices(
    acts,  # type: Union[torch.Tensor, Variable]
    target,  # type: List[List[int]]
    act_lens,  # type: List[int]
    valid_indices,  # type: List[int]
):
    # type: (...) -> (torch.Tensor, List[List[int]], List[int])
    """Copy the CTC inputs without the erroneous samples"""
    if len(valid_indices) == 0:
        return None, [], [], []
    valid_indices = torch.LongTensor(valid_indices)
    aux = acts.new(*valid_indices.shape).long().copy_(valid_indices)
    return (
        # Note: The batch size must be in the second dimension
        torch.index_select(acts, 1, aux),
        [target[i] for i in valid_indices],
        [act_lens[i] for i in valid_indices],
    )


def set_zeros_in_errors(size, input, valid_indices):
    # type: (Sequence[int], torch.Tensor, Sequence[int]) -> torch.Tensor
    """Copy the tensor with zeros in the erroneous indices"""
    if not isinstance(size, (list, tuple)):
        size = list(size)
    valid_indices = (
        torch.cuda.LongTensor(valid_indices)
        if input.is_cuda
        else torch.LongTensor(valid_indices)
    )
    # Note: The batch size must be in the second dimension
    return input.new(*size).zero_().index_copy_(1, valid_indices, input)


def get_valids_and_errors(act_lens, labels):
    # type: (List[int], List[List[int]]) -> (List[int], List[int])
    """Check for sequences which are too short to produce the given
    target, according to CTC model.

    Necessary to avoid potential buffer overflows in CTC.
    """
    assert len(act_lens) == len(labels)

    def count_minimum_frames(y):
        # type: (List[int]) -> int
        return len(y) + sum(y[i] == y[i - 1] for i in range(1, len(y)))

    check = [
        act_lens[i] >= count_minimum_frames(labels[i]) for i in range(len(act_lens))
    ]
    return (
        # Indices of OK samples
        [i for i, valid in enumerate(check) if valid],
        # Indices of the samples with errors regarding their ref length
        [i for i, valid in enumerate(check) if not valid],
    )


class CTCPrepare(Function):
    @staticmethod
    def forward(
        ctx,
        acts,  # type: torch.Tensor
        target,  # type: List[List[int]]
        act_lens,  # type: List[int]
        valid_indices=None,  # type: Optional[List[int]]
    ):
        # type: (...) -> (torch.Tensor, torch.IntTensor * 3)
        """
        Args:
            acts: Contains the output from the network.
                Size seqLength x batchSize x outputDim
            target: Contains all the targets of the batch in
                one sequence. 1 dimensional
            act_lens: Contains the size of each output
                sequence from the network. Size batchSize
            valid_indices: If present, use only these samples to compute
                the CTC loss.
        """
        # Save for backward
        ctx.saved = valid_indices, list(acts.size())

        if valid_indices:
            acts, target, act_lens = copy_valid_indices(
                acts, target, act_lens, valid_indices
            )

        # Prepare tensors of the correct type
        act_lens = torch.IntTensor(act_lens)
        labels = torch.IntTensor(list(itertools.chain.from_iterable(target)))
        label_lens = torch.IntTensor([len(x) for x in target])

        return acts, labels, act_lens, label_lens

    @staticmethod
    def backward(ctx, grad_output, *_):
        valid_indices, original_size = ctx.saved
        return (
            Variable(
                set_zeros_in_errors(original_size, grad_output.data, valid_indices)
            )
            if valid_indices
            else grad_output,
            None,
            None,
            None,
        )


class CTCLoss(Loss):
    """
    Attributes:
        size_average (optional): normalize the loss by the batch size
            (default: `True`)
        length_average (optional): normalize the loss by the total number of frames
            in the batch. If `True`, supersedes `size_average`
            (default: `False`)
    """

    def __init__(self, size_average=True, length_average=False):
        # type: (bool, bool) -> None
        super(CTCLoss, self).__init__()
        self._size_average = size_average
        self._length_average = length_average

    def forward(self, output, target, **kwargs):
        # type: (torch.Tensor, List[List[int]]) -> (FloatScalar, List[int])
        """
        Args:
            output: Size seqLength x outputDim, contains
                the output from the network as well as a list of size
                seqLength containing batch sizes of the sequence
            target: Contains the size of each output
                sequence from the network. Size batchSize
        """
        acts, act_lens = transform_output(output)

        assert act_lens[0] == acts.size(0), "Maximum length does not match"
        assert len(target) == acts.size(1), "Batch size does not match"

        valid_indices, err_indices = get_valids_and_errors(act_lens, target)
        if err_indices:
            if "batch_ids" in kwargs and kwargs["batch_ids"] is not None:
                assert isinstance(kwargs["batch_ids"], (list, tuple))
                err_indices = [kwargs["batch_ids"][i] for i in err_indices]
            _logger.warning(
                "The following samples in the batch were ignored for the loss "
                "computation: {}",
                err_indices,
            )

        if not valid_indices:
            _logger.warning("All samples in the batch were ignored!")
            return None

        acts, labels, act_lens, label_lens = CTCPrepare.apply(
            acts, target, act_lens, valid_indices if err_indices else None
        )

        return _WARP_CTC.apply(
            acts, labels, act_lens, label_lens, self._size_average, self._length_average
        )
