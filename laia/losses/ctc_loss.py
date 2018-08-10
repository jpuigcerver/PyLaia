import itertools
from typing import List, Union, Sequence, Optional

import torch
from torch.autograd import Function
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import laia.common.logging as log
from laia.losses.loss import Loss

try:
    from warpctc_pytorch import _CTC as _WARP_CTC
except ImportError:
    import warnings

    warnings.warn("Missing CTC loss function library")

# TODO(jpuigcerver): Properly tests the logged messages.
_logger = log.get_logger(__name__)


def transform_output(
    output: Union[torch.Tensor, PackedSequence]
) -> (torch.Tensor, List[int]):
    # Size: T x N x D
    if isinstance(output, PackedSequence):
        acts, act_lens = pad_packed_sequence(output)
    elif isinstance(output, torch.Tensor):
        acts, act_lens = output, [output.size(0)] * output.size(1)
    else:
        raise NotImplementedError("Not implemented for type {}".format(type(output)))
    return acts, act_lens


def copy_valid_indices(
    acts: torch.Tensor,
    target: List[List[int]],
    act_lens: List[int],
    valid_indices: List[int],
) -> (Optional[torch.Tensor], List[List[int]], List[int]):
    """Copy the CTC inputs without the erroneous samples"""
    if len(valid_indices) == 0:
        return None, [], []
    valid_indices = torch.tensor(valid_indices, device=acts.device)
    return (
        # Note: The batch size must be in the second dimension
        torch.index_select(acts, 1, valid_indices),
        [target[i] for i in valid_indices],
        [act_lens[i] for i in valid_indices],
    )


def set_zeros_in_errors(
    size: Sequence[int], input: torch.Tensor, valid_indices: List[int]
) -> torch.Tensor:
    """Copy the tensor with zeros in the erroneous indices"""
    valid_indices = torch.tensor(valid_indices, device=input.device)
    # Note: The batch size must be in the second dimension
    return input.new_zeros(size).index_copy_(1, valid_indices, input)


def get_valids_and_errors(
    act_lens: List[int], labels: List[List[int]]
) -> (List[int], List[int]):
    """Check for sequences which are too short to produce the given
    target, according to CTC model.

    Necessary to avoid potential buffer overflows in CTC.
    """
    assert len(act_lens) == len(labels)

    def count_minimum_frames(y: List[int]) -> int:
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
        ctx: torch.autograd.function.BackwardCFunction,
        acts: torch.Tensor,
        target: List[List[int]],
        act_lens: List[int],
        valid_indices: Optional[List[int]] = None,
    ) -> (torch.Tensor, torch.IntTensor, torch.IntTensor, torch.IntTesor):
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
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,
        grad_output: torch.Tensor,
        *_: None
    ) -> (torch.Tensor, None, None, None):
        valid_indices, original_size = ctx.saved
        return (
            set_zeros_in_errors(original_size, grad_output, valid_indices)
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

    def __init__(self, size_average: bool = True, length_average: bool = False) -> None:
        super().__init__()
        self._size_average = size_average
        self._length_average = length_average

    def forward(
        self, output: torch.Tensor, target: List[List[int]], **kwargs: dict
    ) -> (torch.FloatTensor, List[int]):
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
            if kwargs.get("batch_ids", None) is not None:
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
