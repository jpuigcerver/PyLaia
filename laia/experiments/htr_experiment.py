from __future__ import absolute_import

from typing import Callable, Optional, Sequence, List

from laia.common.logging import get_logger
from laia.decoders import CTCGreedyDecoder
from laia.engine import Trainer, Evaluator
from laia.engine.engine import EPOCH_END, ITER_END
from laia.experiments import Experiment
from laia.hooks import action
from laia.losses import CTCLoss
from laia.meters import SequenceErrorMeter
from laia.utils.char_to_word_seq import char_to_word_seq

_logger = get_logger(__name__)


def batch_char_to_word_seq(batch_sequence_of_characters, delimiters):
    return [
        ["_".join(map(str, seq)) for seq in char_to_word_seq(batch, delimiters)]
        for batch in batch_sequence_of_characters
    ]


class HTRExperiment(Experiment):
    def __init__(
        self,
        train_engine,  # type: Trainer
        valid_engine=None,  # type: Optional[Evaluator]
        check_valid_hook_when=EPOCH_END,  # type: Optional[str]
        valid_hook_condition=None,  # type: Optional[Callable]
        word_delimiters=None,  # type: Optional[Sequence]
        summary_order=(
            "Epoch",
            "TR Loss",
            "VA Loss",
            "TR CER",
            "VA CER",
            "TR WER",
            "VA WER",
            "TR Time",
            "VA Time",
            "Memory",
        ),  # type: Sequence[str]
    ):
        # type: (...) -> None
        super(HTRExperiment, self).__init__(
            train_engine,
            valid_engine=valid_engine,
            check_valid_hook_when=check_valid_hook_when,
            valid_hook_condition=valid_hook_condition,
            summary_order=summary_order,
        )
        self._word_delimiters = word_delimiters

        # If the trainer was created without any criterion,
        # or it is not the CTCLoss, set it properly.
        if not self._tr_engine.criterion:
            self._tr_engine.criterion = CTCLoss()
        elif not isinstance(self._tr_engine.criterion, CTCLoss):
            self.logger.warn("Overriding the criterion of the trainer to CTC.")
            self._tr_engine.criterion = CTCLoss()

        self._ctc_decoder = CTCGreedyDecoder()
        self._tr_cer = SequenceErrorMeter()
        self._tr_wer = SequenceErrorMeter()

        self._tr_engine.add_hook(ITER_END, self._train_update_meters)

        if self._va_engine:
            self._va_cer = SequenceErrorMeter()
            self._va_wer = SequenceErrorMeter()

            self._va_engine.add_hook(ITER_END, self._valid_update_meters)
        else:
            self._va_cer = None
            self._va_wer = None

    def train_cer(self):
        return self._tr_cer

    def valid_cer(self):
        return self._va_cer

    def train_wer(self):
        return self._tr_wer

    def valid_wer(self):
        return self._va_wer

    def set_word_delimiters(self, delimiters):
        self._word_delimiters = delimiters

    @action
    def train_reset_meters(self):
        super(HTRExperiment, self).train_reset_meters()
        self._tr_cer.reset()
        self._tr_wer.reset()

    @action
    def valid_reset_meters(self):
        super(HTRExperiment, self).valid_reset_meters()
        self._va_cer.reset()
        self._va_wer.reset()

    @action
    def _train_update_meters(self, batch_loss, batch_output, batch_target):
        self._tr_loss.add(batch_loss)

        # Compute CER
        batch_decode = self._ctc_decoder(batch_output)
        self._tr_cer.add(batch_target, batch_decode)

        # Compute WER, if word delimiters are given
        if self._word_delimiters is not None:
            decode_words = batch_char_to_word_seq(batch_decode, self._word_delimiters)
            target_words = batch_char_to_word_seq(batch_target, self._word_delimiters)
            self._tr_wer.add(target_words, decode_words)

        # Stop timer to avoid including extra costs
        self._tr_timer.stop()

    @action
    def _valid_update_meters(self, batch, batch_output, batch_target):
        batch_loss = self._tr_engine.compute_loss(batch, batch_output, batch_target)
        if batch_loss is not None:
            self._va_loss.add(batch_loss)

        # Compute CER
        batch_decode = self._ctc_decoder(batch_output)
        self._va_cer.add(batch_target, batch_decode)

        # Compute WER, if word delimiters are given
        if self._word_delimiters is not None:
            decode_words = batch_char_to_word_seq(batch_decode, self._word_delimiters)
            target_words = batch_char_to_word_seq(batch_target, self._word_delimiters)
            self._va_wer.add(target_words, decode_words)

        # Stop timer to avoid including extra costs
        self._va_timer.stop()

    def epoch_summary(self, summary_order=None):
        # type: (Optional[Sequence[str]]) -> List[dict]
        summary = super(HTRExperiment, self).epoch_summary(summary_order=summary_order)
        summary.append(
            dict(label="TR CER", format="{.value:5.1%}", source=self._tr_cer)
        )
        if self._va_engine:
            summary.append(
                dict(label="VA CER", format="{.value:5.1%}", source=self._va_cer)
            )
        if self._word_delimiters is not None:
            summary.append(
                dict(label="TR WER", format="{.value:5.1%}", source=self._tr_wer)
            )
        if self._va_engine and self._word_delimiters is not None:
            summary.append(
                dict(label="VA WER", format="{.value:5.1%}", source=self._va_wer)
            )
        try:
            return sorted(summary, key=lambda d: summary_order.index(d["label"]))
        except (AttributeError, ValueError) as e:
            _logger.debug("Could not sort the summary. Reason: {}", e)
            return summary

    def state_dict(self):
        # type: () -> dict
        state = super(HTRExperiment, self).state_dict()
        for k, v in (
            ("tr_cer", self._tr_cer),
            ("va_cer", self._va_cer),
            ("tr_wer", self._tr_wer),
            ("va_wer", self._va_wer),
        ):
            if hasattr(v, "state_dict"):
                state[k] = v.state_dict()
        return state

    def load_state_dict(self, state):
        # type: (dict) -> None
        if state is None:
            return
        super(HTRExperiment, self).load_state_dict(state)
        for k, v in (
            ("tr_cer", self._tr_cer),
            ("va_cer", self._va_cer),
            ("tr_wer", self._tr_wer),
            ("va_wer", self._va_wer),
        ):
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(state[k])
