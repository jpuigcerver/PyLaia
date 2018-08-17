from typing import Any, Dict

import laia.common.logging as log

_logger = log.get_logger(__name__)


class Meter:
    """A meter returns its :attr:`~.Meter.value` when it is called

    Meters are used for measuring times, computing running averages, and
    many other metrics.

    If the value of the meter cannot be read (Returns `None` or
    causes an exception when its called), we assume that the meter has
    not produced a value yet and returns None.

    The number of exceptions will be recorded and an error will be logged
    when `exceptions_threshold` exceptions are caught.

    See for instance:
      - :class:`~laia.meters.RunningAverageMeter`
      - :class:`~laia.meters.SequenceErrorMeter`
      - :class:`~laia.meters.TimeMeter`

    Arguments:
        exceptions_threshold (int): number of exceptions to catch from the
            meter before logging. (default: 5)
    """

    def __init__(self, exceptions_threshold: int = 5) -> None:
        self._exceptions = 0
        self._exceptions_threshold = exceptions_threshold

    @property
    def value(self):
        """Access the latest value of the meter."""
        raise NotImplementedError("This method should be overridden.")

    def __call__(self) -> Any:
        # Try to get the meter's last value, if some exception occurs,
        # we assume that the meter has not produced any value yet, and
        # we do not trigger.
        value = None
        try:
            value = self.value
            assert value is not None, "Meter returned None"
        except Exception:
            self._exceptions += 1
            if self._exceptions % self._exceptions_threshold == 0:
                _logger.warn(
                    "No value fetched from meter after a while "
                    "({} exceptions like this occurred so far)",
                    self._exceptions,
                )
        return value

    def state_dict(self) -> Dict:
        return {"exceptions": self._exceptions}

    def load_state_dict(self, state: Dict) -> None:
        self._exceptions = state["exceptions"]
