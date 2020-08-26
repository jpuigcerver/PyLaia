from typing import Any


class Meter:
    """A meter returns its :attr:`~.Meter.value` when it is called."""

    @property
    def value(self):
        """Access the latest value of the meter."""
        raise NotImplementedError

    def __call__(self) -> Any:
        value = self.value
        assert value is not None, "Meter returned None"
        return value
