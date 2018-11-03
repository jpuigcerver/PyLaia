from __future__ import absolute_import

import itertools
from typing import List, Union, Sequence

import torch
from torch.autograd import Function
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import laia.common.logging as log
from laia.losses.loss import Loss

try:
    from torch_baidu_ctc import ctc_loss
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
    elif isinstance(output, torch.Tensor):
        acts, act_lens = output, [output.size(0)] * output.size(1)
    else:
        raise NotImplementedError("Not implemented for type {}".format(type(output)))
    return acts, act_lens


def copy_valid_indices(
    acts,  # type: torch.Tensor
    target,  # type: List[List[int]]
    act_lens,  # type: List[int]
    valid_indices,  # type: List[int]
):
    # type: (...) -> (torch.Tensor, List[List[int]], List[int])
    """Copy the CTC inputs without the erroneous samples"""
    if len(valid_indices) == 0:
        return None, [], [], []
    valid_indices = torch.tensor(valid_indices, device=acts.device)
    return (
        # Note: The batch size must be in the second dimension
        torch.index_select(acts, 1, valid_indices),
        [target[i] for i in valid_indices],
        [act_lens[i] for i in valid_indices],
    )


def set_zeros_in_errors(size, input, valid_indices):
    # type: (Sequence[int], torch.Tensor, Sequence[int]) -> torch.Tensor
    """Copy the tensor with zeros in the erroneous indices"""
    valid_indices = torch.tensor(valid_indices, device=input.device)
    # Note: The batch size must be in the second dimension
    return input.new_zeros(size).index_copy_(1, valid_indices, input)


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
        act_lens = torch.tensor(act_lens, dtype=torch.int32, device=torch.device("cpu"))
        labels = torch.tensor(
            list(itertools.chain.from_iterable(target)),
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        label_lens = torch.tensor(
            [len(x) for x in target], dtype=torch.int32, device=torch.device("cpu")
        )

        return acts, labels, act_lens, label_lens

    @staticmethod
    def backward(ctx, grad_output, *_):
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
      reduction (string, optional): Specifies the type of reduction to
        perform on the minibatch costs: 'none' | 'mean' | 'sum'.
        With 'none': no reduction is done and a tensor with the cost of each
        sample in the minibatch is returned,
        'mean': the mean of all costs in the minibatch is returned,
        'sum': the sum of all costs in the minibatch is returned.
        Default: 'sum'.
      average_frames (bool, optional): Specifies whether the loss of each
        sample should be divided by its number of frames. Default: ``False''.
    """

    def __init__(self, reduction="mean", average_frames=False):
        # type: (bool, bool) -> None
        super(CTCLoss, self).__init__()
        self._reduction = reduction
        self._average_frames = average_frames

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

        # TODO(jpuigcerver): We need to change this because CTCPrepare.apply
        # will set requires_grad of *all* outputs to True if *any* of the
        # inputs requires_grad is True.
        acts, labels, act_lens, label_lens = CTCPrepare.apply(
            acts, target, act_lens, valid_indices if err_indices else None
        )

        # TODO(jpuigcerver): Remove the detach() once the previous TODO is
        # fixed.
        return ctc_loss(
            acts=acts,
            labels=labels.detach(),
            acts_lens=act_lens.detach(),
            labels_lens=label_lens.detach(),
            reduction=self._reduction,
            average_frames=self._average_frames,
        )
