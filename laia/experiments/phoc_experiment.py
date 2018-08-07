from __future__ import absolute_import

from typing import Optional, Callable

import torch
from torch.nn.functional import sigmoid

import laia.common.logging as log
from laia.engine import Evaluator, Trainer
from laia.engine.engine import ITER_END, EPOCH_END
from laia.engine.feeders import ImageFeeder, ItemFeeder, PHOCFeeder, VariableFeeder
from laia.experiments import Experiment
from laia.hooks import action
from laia.meters import PairwiseAveragePrecisionMeter, Meter

_logger = log.get_logger(__name__)


class PHOCExperiment(Experiment):
    r"""Wrapper to perform KWS experiments with PHOC networks."""

    def __init__(
        self,
        symbols_table,
        phoc_levels,
        train_engine,  # type: Trainer
        valid_engine=None,  # type: Optional[Evaluator]
        check_valid_hook_when=EPOCH_END,  # type: Optional[str]
        valid_hook_condition=None,  # type: Optional[Callable]
        gpu=0,
        exclude_labels=None,
        ignore_missing=False,
        use_new_phoc=False,
        summary_order=(
            "Epoch",
            "TR Loss",
            "VA Loss",
            "VA gAP",
            "VA mAp",
            "TR Time",
            "VA Time",
            "Memory",
        ),  # type: Sequence[str]
    ):
        # type: (...) -> None
        super(PHOCExperiment, self).__init__(
            train_engine,
            valid_engine=valid_engine,
            check_valid_hook_when=check_valid_hook_when,
            valid_hook_condition=valid_hook_condition,
            summary_order=summary_order,
        )

        # If the trainer was created without any criterion, set it properly.
        if not self._tr_engine.criterion:
            self._tr_engine.criterion = torch.nn.BCEWithLogitsLoss()

        # Set trainer's batch_input_fn and batch_target_fn if not already set.
        if not self._tr_engine.batch_input_fn:
            self._tr_engine.set_batch_input_fn(
                ImageFeeder(
                    device=gpu,
                    keep_padded_tensors=False,
                    parent_feeder=ItemFeeder("img"),
                )
            )
        if not self._tr_engine.batch_target_fn:
            self._tr_engine.set_batch_target_fn(
                VariableFeeder(
                    device=gpu,
                    parent_feeder=PHOCFeeder(
                        syms=symbols_table,
                        levels=phoc_levels,
                        ignore_missing=ignore_missing,
                        new_phoc=use_new_phoc,
                        parent_feeder=ItemFeeder("txt"),
                    ),
                )
            )

        self._tr_engine.add_hook(ITER_END, self._train_accumulate_loss)

        if valid_engine:
            # Set batch_input_fn and batch_target_fn if not already set.
            if not self._va_engine.batch_input_fn:
                self._va_engine.set_batch_input_fn(self._tr_engine.batch_input_fn)
            if not self._va_engine.batch_target_fn:
                self._va_engine.set_batch_target_fn(self._tr_engine.batch_target_fn)

            self._va_ap = PairwiseAveragePrecisionMeter(
                metric="braycurtis",
                ignore_singleton=True,
                exclude_labels=exclude_labels,
            )

            self._va_engine.add_hook(ITER_END, self._valid_accumulate_loss)
        else:
            self._va_ap = None

    def valid_ap(self):
        # type: () -> Meter
        return self._va_ap

    @action
    def valid_reset_meters(self):
        super(PHOCExperiment, self).valid_reset_meters()
        self._va_ap.reset()

    @action
    def _train_accumulate_loss(self, batch_loss):
        self._tr_loss.add(batch_loss)
        # Stop timer to avoid including extra costs
        self._tr_timer.stop()

    @action
    def _valid_accumulate_loss(self, batch, batch_output, batch_target):
        batch_loss = self._tr_engine.criterion(batch_output, batch_target)
        self._va_loss.add(batch_loss)

        batch_output_phoc = sigmoid(batch_output.data)
        self._va_ap.add(
            batch_output_phoc.cpu().numpy(), ["".join(w) for w in batch["txt"]]
        )
        self._va_timer.stop()

    def epoch_summary(self, summary_order=None):
        # type: (Optional[Sequence[str]]) -> List[dict]
        summary = super(PHOCExperiment, self).epoch_summary(summary_order=summary_order)
        if self._va_engine:
            summary.append(
                dict(label="VA gAP", format="{.value[0]:5.1%}", source=self._va_ap)
            )
            summary.append(
                dict(label="VA mAP", format="{.value[1]:5.1%}", source=self._va_ap)
            )
        try:
            return sorted(summary, key=lambda d: summary_order.index(d["label"]))
        except (AttributeError, ValueError) as e:
            _logger.debug("Could not sort the summary. Reason: {}", e)
            return summary

    def state_dict(self):
        # type: () -> dict
        state = super(PHOCExperiment, self).state_dict()
        if hasattr(self._va_ap, "state_dict"):
            state["va_ap"] = self._va_ap.state_dict()
        return state

    def load_state_dict(self, state):
        # type: (dict) -> None
        if state is None:
            return
        super(PHOCExperiment, self).load_state_dict(state)
        if hasattr(self._va_ap, "load_state_dict"):
            self._va_ap.load_state_dict(state["va_ap"])
