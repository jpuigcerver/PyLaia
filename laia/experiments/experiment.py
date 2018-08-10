from collections import Sequence
from typing import Callable, Optional, List

import laia
from laia.common.logging import get_logger
from laia.engine.engine import Evaluator, EPOCH_START, EPOCH_END, Engine
from laia.hooks import action
from laia.meters import RunningAverageMeter, TimeMeter, MemoryMeter, Meter

_logger = get_logger(__name__)


class Experiment:
    def __init__(
        self,
        train_engine: laia.engine.Trainer,
        valid_engine: Optional[Evaluator] = None,
        check_valid_hook_when: Optional[str] = EPOCH_END,
        valid_hook_condition: Optional[Callable] = None,
        summary_order: Sequence[str] = (
            "Epoch",
            "TR Loss",
            "VA Loss",
            "TR Time",
            "VA Time",
            "Memory",
        ),
    ) -> None:
        self._tr_engine = train_engine
        self._va_engine = valid_engine

        self._tr_timer = TimeMeter()
        self._tr_loss = RunningAverageMeter()

        self._tr_engine.add_hook(EPOCH_START, self.train_reset_meters)

        if self._va_engine:
            self._va_timer = TimeMeter()
            self._va_loss = RunningAverageMeter()

            self._va_engine.add_hook(EPOCH_START, self.valid_reset_meters)
            self._tr_engine.add_evaluator(
                self._va_engine,
                when=check_valid_hook_when,
                condition=valid_hook_condition,
            )
        else:
            self._va_timer = None
            self._va_loss = None

        self._summary_order = summary_order
        self._summary = None
        self._tr_engine.add_hook(EPOCH_END, self._log_epoch_summary)

    def train_timer(self) -> Meter:
        return self._tr_timer

    def valid_timer(self) -> Meter:
        return self._va_timer

    def train_loss(self) -> Meter:
        return self._tr_loss

    def valid_loss(self) -> Meter:
        return self._va_loss

    def run(self):
        self._summary = self.epoch_summary(self._summary_order)
        self._tr_engine.run()
        return self

    @action
    def train_reset_meters(self):
        self._tr_timer.reset()
        self._tr_loss.reset()

    @action
    def valid_reset_meters(self):
        self._va_timer.reset()
        self._va_loss.reset()

    def epoch_summary(
        self, summary_order: Optional[Sequence[str]] = None
    ) -> List[dict]:
        summary = [
            dict(label="Epoch", format="{:4d}"),
            dict(label="TR Loss", format="{.value[0]:.3e}", source=self._tr_loss),
            dict(label="TR Time", format="{.value:.2f}s", source=self._tr_timer),
            dict(label="Memory", format="{.value}", source=MemoryMeter()),
        ]
        if self._va_engine:
            summary.append(
                dict(label="VA Loss", format="{.value[0]:.3e}", source=self._va_loss)
            )
            summary.append(
                dict(label="VA Time", format="{.value:.2f}s", source=self._va_timer)
            )
        try:
            return sorted(summary, key=lambda d: summary_order.index(d["label"]))
        except (AttributeError, ValueError) as e:
            _logger.debug("Could not sort the summary. Reason: {}", e)
            return summary

    @action
    def _log_epoch_summary(self, epoch: int) -> None:
        for item in self._summary:
            if item["label"] == "Epoch":
                item["source"] = epoch
        parsed_summary = [
            "{} = {}".format(x["label"], x["format"].format(x["source"]))
            for x in self._summary
        ]
        _logger.info(", ".join(parsed_summary))

    def state_dict(self) -> dict:
        return {
            k: v.state_dict() if hasattr(v, "state_dict") else None
            for k, v in (
                ("tr_engine", self._tr_engine),
                ("tr_loss", self._tr_loss),
                ("va_loss", self._va_loss),
            )
        }

    def load_state_dict(self, state: dict) -> None:
        if state is None:
            return
        for k, v in (
            ("tr_engine", self._tr_engine),
            ("tr_loss", self._tr_loss),
            ("va_loss", self._va_loss),
        ):
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(state[k])

    @staticmethod
    def get_model_state_dict(state: dict) -> dict:
        return Engine.get_model_state_dict(state["tr_engine"])
