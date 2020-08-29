import itertools
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import laia.common.logging as log
from laia.losses.loss import Loss

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


def get_valids_and_errors(
    act_lens: List[int], labels: List[List[int]]
) -> Tuple[List[int], List[int]]:
    """Check for sequences which are too short to produce the given
    target, according to CTC model. These could produce infinite
    losses or potential buffer overflows in CTC.
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
    """

    def __init__(self, reduction="mean", average_frames=False, blank=0):
        super().__init__()
        self._reduction = reduction
        self._average_frames = average_frames
        self._blank = blank

    @property
    def average_frames(self):
        return self._average_frames

    @average_frames.setter
    def average_frames(self, value):
        assert value in (True, False)
        self._average_frames = value

    @property
    def blank(self):
        return self._blank

    @blank.setter
    def blank(self, value):
        assert isinstance(value, int) and value >= 0
        self._blank = value

    @property
    def reduction(self):
        return self._reduction

    @reduction.setter
    def reduction(self, value):
        self._reduction = value

    def forward(
        self, output: torch.Tensor, target: List[List[int]], **kwargs: Dict
    ) -> torch.Tensor:
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
                "The following samples in the batch were presumably ignored "
                "for the loss computation: {}",
                err_indices,
            )
        if not valid_indices:
            _logger.warning("All samples in the batch were ignored!")

        # Prepare tensors of the correct type
        acts = torch.nn.functional.log_softmax(acts, dim=-1)
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

        with torch.backends.cudnn.flags(enabled=False):
            losses = torch.nn.functional.ctc_loss(
                log_probs=acts,
                targets=labels.to(acts.device),
                input_lengths=act_lens,
                target_lengths=label_lens,
                blank=self._blank,
                reduction="none",
                zero_infinity=True,
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
            raise ValueError(f"Reduction {repr(self._reduction)} not supported!")
