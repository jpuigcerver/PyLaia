from __future__ import absolute_import

import enum
import itertools
from typing import List, Union, Sequence

import torch
from torch.autograd import Function
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import laia.common.logging as log
from laia.losses.loss import Loss

# Try to import torch_baidu_ctc
try:
    import torch_baidu_ctc
except ImportError:
    torch_baidu_ctc = None

FloatScalar = Union[float, torch.FloatTensor]

_logger = log.get_logger(__name__)


_WARNING_BAIDU_NOT_AVAILABLE = (
    "Baidu's CTC implementation is not available, PyTorch's own implementation "
    "will be used instead. If you want to use Baidu's, please install the package "
    "`torch_baidu_ctc`."
)

_WARNING_BAIDU_WRONG = (
    "You are using Baidu's implementation of the CTC loss. Be aware that this implementation "
    "is WRONG. During the forward pass, their implementation performs an implicit softmax "
    "normalization of the input activations. However, this operation is not taken into account "
    "during the computation of the gradients w.r.t. the original activations."
)

_WARNING_PYTORCH_LOGSOFTMAX = (
    "You are using PyTorch's CTC loss, but add_logsoftmax is False. Notice that the activations "
    "must be properly normalized (i.e. the log-sum-exp equals to 0.0) for a correct behavior."
)


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
        if isinstance(act_lens, torch.Tensor):
            act_lens = act_lens.detach().to(torch.int32).cpu()
        else:
            act_lens = torch.tensor(act_lens, dtype=torch.int32, device="cpu")
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


# Class and functions to represent and set the default CTC implementation to use.


@enum.unique
class CTCLossImpl(enum.Enum):
    PYTORCH = enum.auto()
    BAIDU = enum.auto()


_default_implementation = CTCLossImpl.PYTORCH


def get_default_implementation():
    global _default_implementation
    return _default_implementation


def set_default_implementation(implementation):
    assert isinstance(implementation, CTCLossImpl)
    global _default_implementation
    _default_implementation = implementation


# Functions to control whether or not to add a logsoftmax operation by default.

_default_add_logsoftmax = True


def get_default_add_logsoftmax():
    global _default_add_logsoftmax
    return _default_add_logsoftmax


def set_default_add_logsoftmax(add_logsoftmax):
    assert add_logsoftmax in (True, False)
    global _default_add_logsoftmax
    _default_add_logsoftmax = add_logsoftmax


class CTCLoss(Loss):
    """
    Attributes:
      reduction (string, optional): Specifies the type of reduction to
        perform on the minibatch costs: 'none' | 'mean' | 'sum'.
        With 'none': no reduction is done and a tensor with the cost of each
        example in the minibatch is returned,
        'mean': the mean of the per-example losses is returned,
        'sum': the sum of all per-example losses is returned.
        Default: 'mean'.
      average_frames (bool, optional): Specifies whether the loss of each
        sample should be divided by its number of frames. Default: ``False''.
      blank (integer, optional): Index of the blank label. Default: 0.
      add_logsoftmax (bool, optional): Whether or not to add an implicit
        logsoftmax operation on the activations before the loss.
        If not specified, uses the default value given by
        `ctc_loss.get_default_add_logsoftmax()`.
      implementation (integer, optional): Specify the implementation of the
        CTCLoss to use. If not specified, uses the default implementation
        given by `ctc_loss.get_default_implementation()`.
    """

    def __init__(
        self,
        reduction="mean",
        average_frames=False,
        blank=0,
        add_logsoftmax=None,
        implementation=None,
    ):
        # type: (bool, bool) -> None
        super(CTCLoss, self).__init__()
        self._reduction = reduction
        self._average_frames = average_frames
        self._blank = blank
        self._add_logsoftmax = None
        self._implementation = None

        self.add_logsoftmax = add_logsoftmax
        self.implementation = implementation

    @property
    def add_logsoftmax(self):
        return self._add_logsoftmax

    @add_logsoftmax.setter
    def add_logsoftmax(self, value):
        if value is None:
            value = get_default_add_logsoftmax()

        self._add_logsoftmax = value
        if self._implementation == CTCLossImpl.PYTORCH and not self._add_logsoftmax:
            _logger.warning(_WARNING_PYTORCH_LOGSOFTMAX)

    @property
    def average_frames(self):
        return self._average_frames

    @average_frames.setter
    def average_frames(self, value):
        assert value in (True, False)
        self._average_frames = average_frames

    @property
    def blank(self):
        return self._blank

    @blank.setter
    def blank(self, value):
        assert isinstance(value, int) and value >= 0
        self._blank = value

    @property
    def implementation(self):
        return self._implementation

    @implementation.setter
    def implementation(self, value):
        if value is None:
            value = get_default_implementation()

        if value not in (CTCLossImpl.PYTORCH, CTCLossImpl.BAIDU):
            raise ValueError("Unknown CTC implementation: {!r}".format(value))
        elif value == CTCLossImpl.BAIDU:
            if torch_baidu_ctc is None:
                _logger.warning(_WARNING_BAIDU_NOT_AVAILABLE)
                value = CTCLossImpl.PYTORCH
            else:
                _logger.warning(_WARNING_BAIDU_WRONG)

        if value == CTCLossImpl.PYTORCH and not self._add_logsoftmax:
            _logger.warning(_WARNING_PYTORCH_LOGSOFTMAX)

        self._implementation = value
        return self

    @property
    def reduction(self):
        return self._reduction

    @reduction.setter
    def reduction(self, value):
        self._reduction = value

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
        labels = labels.detach()
        act_lens = act_lens.detach()
        label_lens = label_lens.detach()

        if self._add_logsoftmax:
            acts = torch.nn.functional.log_softmax(acts, dim=-1)

        if self._implementation == CTCLossImpl.PYTORCH:
            losses = torch.nn.functional.ctc_loss(
                log_probs=acts,
                targets=labels,
                input_lengths=act_lens,
                target_lengths=label_lens,
                blank=self._blank,
                reduction="none",
            )

            if self._average_frames:
                losses = losses / act_lens.to(losses)

            if self._reduction == "none":
                return losses
            elif self._reduction == "mean":
                return losses.mean()
            elif self._reduction == "sum":
                return losses.sum()
            else:
                raise ValueError(
                    "Reduction {!r} not supported!".format(self._reduction)
                )
        elif self._implementation == CTCLossImpl.BAIDU:
            return torch_baidu_ctc.ctc_loss(
                acts=acts,
                labels=labels,
                acts_lens=act_lens,
                labels_lens=label_lens,
                reduction=self._reduction,
                average_frames=self._average_frames,
            )
        else:
            raise ValueError(
                "Unknown CTC implementation: {!r}".format(self._implementation)
            )
