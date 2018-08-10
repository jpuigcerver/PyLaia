from typing import Tuple, Callable, Any as AnyT

from laia.common.logging import get_logger, DEBUG, INFO, ERROR

_logger = get_logger(__name__)


class Condition(object):
    """Conditions are objects that when called return either `True` or `False`.
    Typically used inside of Hooks to trigger an action

    Arguments:
        obj (Callable): obj from which a value will be retrieved
        key (Any, optional): Get this key from the obj after being called.
            Useful when the obj() returns a tuple/list/dict. (default: None)
    """

    def __init__(self, obj, key=None):
        # type: (Callable, AnyT) -> None
        self._obj = obj
        self._key = key

    def __call__(self):
        raise NotImplementedError

    def _process_value(self):
        value = self._obj()
        if value is None:
            # An exception happened during the computation
            return None
        return value if self._key is None else value[self._key]

    def state_dict(self):
        return {
            "obj": self._obj.state_dict() if hasattr(self._obj, "state_dict") else None
        }

    def load_state_dict(self, state):
        if hasattr(self._obj, "load_state_dict"):
            self._obj.load_state_dict(state["obj"])


class LoggingCondition(Condition):
    def __init__(self, obj, key, logger, name=None):
        # type: (obj, AnyT, Logger, str) -> None
        super(LoggingCondition, self).__init__(obj, key)
        self._logger = logger
        self._name = name

    def __call__(self):
        raise NotImplementedError

    @property
    def logger(self):
        return self._logger

    @property
    def name(self):
        return self._name

    def log(self, level, msg, *args, **kwargs):
        self._logger.log(
            level,
            'Condition "{}": {}'.format(self.name, msg) if self.name else msg,
            *args,
            **kwargs
        )

    def debug(self, msg, *args, **kwargs):
        self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(INFO, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(ERROR, msg, *args, **kwargs)


class Not(object):
    """True when the given condition is false.

    Arguments:
        condition (:obj:`~Condition`): condition to be negated
    """

    def __init__(self, condition):
        # type: (Callable) -> None
        assert callable(condition)
        self._condition = condition

    def __call__(self):
        return not self._condition()


class MultinaryCondition(object):
    """Base class for operators involving an arbitrary number of conditions.

    Arguments:
        conditions (:obj:`~Condition`): parameters of the operation.
    """

    def __init__(self, *conditions):
        # type: (Tuple[Callable]) -> None
        assert all(callable(c) for c in conditions)
        self._conditions = conditions

    def __call__(self):
        raise NotImplementedError


class Any(MultinaryCondition):
    """Returns `True` if any of the given `conditions` returns `True`."""

    def __call__(self):
        return any(c() for c in self._conditions)


class All(MultinaryCondition):
    """Returns `True` if all of the given `conditions` return `True`."""

    def __call__(self):
        return all(c() for c in self._conditions)
