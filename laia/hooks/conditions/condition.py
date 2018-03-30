from typing import Tuple

from laia.hooks.meters import Meter
from laia.logging import get_logger, DEBUG, INFO, ERROR

_logger = get_logger(__name__)


class Condition(object):
    """Conditions are objects that when called return either `True` or `False`.

    Typically used for early stopping, creating checkpoints of the
    model during training, logging events, etc.

    They may have a name, which may be useful for logging.

    Arguments:
        name (str): name for the condition. (default: `None`)
    """

    def __init__(self, name=None):
        # type: (str) -> None
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self):
        return False


class LoggingCondition(Condition):
    def __init__(self, logger, name=None):
        # type: (Logger, str) -> None
        super(LoggingCondition, self).__init__(name=name)
        self._logger = logger

    @property
    def logger(self):
        return self._logger

    def log(self, level, msg, *args, **kwargs):
        self._logger.log(
            level,
            'Condition "{}": {}'.format(self.name, msg) if self.name else msg,
            *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(INFO, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(ERROR, msg, *args, **kwargs)


class ConditionFromMeter(LoggingCondition):
    """Base class for conditions based on the value of a :class:`Meter`.

    If the value of the meter cannot be read (`Meter.value` returns `None` or
    causes an exception), we assume that the meter has not produced a value
    yet and the trigger returns false.

    The number of exceptions will be recorded and an error will be logged
    when `num_exceptions_threshold` are caught.

    Arguments:
        meter (:obj:`Meter`): meter to monitor.
        meter_key (Any): if the value returned by the meter is a tuple, list
            or dictionary, use this key to get the specific value.
            (default: None)
        name (str): name of the trigger (default: None).
        num_exceptions_threshold (int): number of exceptions to catch from the
            meter before logging. (default: 5)
    """

    def __init__(self, meter, meter_key=None, name=None,
                 num_exceptions_threshold=5):
        # type: (Meter, str, str, int) -> None
        super(ConditionFromMeter, self).__init__(_logger, name)
        self._meter = meter
        self._meter_key = meter_key
        self._num_exceptions = 0
        self._num_exceptions_threshold = num_exceptions_threshold

    @property
    def meter(self):
        return self._meter

    def _process_value(self, last_value):
        raise NotImplementedError('This method should be implemented')

    def __call__(self):
        # Try to get the meter's last value, if some exception occurs,
        # we assume that the meter has not produced any value yet, and
        # we do not trigger.
        try:
            last_value = self._meter.value
            assert last_value is not None, 'Meter returned None'
        except Exception:
            self._num_exceptions += 1
            if self._num_exceptions % self._num_exceptions_threshold == 0:
                self.logger.warn(
                    'No value fetched from meter after a while '
                    '({} exceptions like this occurred so far)',
                    self._num_exceptions)
            return False

        if self._meter_key is not None:
            last_value = last_value[self._meter_key]

        return self._process_value(last_value)


class Not(Condition):
    """True when the given condition is false.

    Arguments:
        condition (:obj:`~Condition`): condition to be negated
    """

    def __init__(self, condition):
        # type: (Condition) -> None
        assert isinstance(condition, Condition)
        super(Not, self).__init__()
        self._condition = condition

    def __call__(self):
        return not self._condition()


class MultinaryCondition(Condition):
    """Base class for operators involving an arbitrary number of conditions.

    Arguments:
        conditions (:obj:`~Condition`): parameters of the operation.
    """

    def __init__(self, *conditions):
        # type: (Tuple[Condition]) -> None
        assert all([isinstance(c, Condition) for c in conditions])
        super(MultinaryCondition, self).__init__()
        self._conditions = conditions

    def __call__(self):
        raise NotImplemented


class Any(MultinaryCondition):
    """Returns `True` if any of the given `conditions` returns `True`.

    Arguments:
        conditions (:obj:`~Condition`): parameters of the operation.
    """

    def __init__(self, *conditions):
        # type: (Tuple[Condition]) -> None
        super(Any, self).__init__(*conditions)

    def __call__(self):
        return any([c() for c in self._conditions])


class All(MultinaryCondition):
    """Returns `True` if all of the given `conditions` return `True`.

    Arguments:
        conditions (:obj:`~Condition`): parameters of the operation.
    """

    def __init__(self, *conditions):
        # type: (Tuple[Condition]) -> None
        super(All, self).__init__(*conditions)

    def __call__(self):
        return all([c() for c in self._conditions])
