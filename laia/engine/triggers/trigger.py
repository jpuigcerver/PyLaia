from logging import Logger

import typing
from typing import Dict, Tuple


class Trigger(object):
    """
    Triggers are objects that when called return either `True` or `False`.

    Triggers are typically used for early stopping, creating checkpoints of the
    model during training, logging events, etc.

    Triggers may have a name, which may be useful for logging.

    Arguments:
        name (str): name for the trigger. (default: None)
    """

    def __init__(self, name=None):
        # type: (str) -> None
        self._name = name

    @property
    def name(self):
        """Name of the trigger"""
        return self._name

    def __call__(self):
        return False


class LoggedTrigger(Trigger):
    def __init__(self, logger, name=None):
        # type: (Logger, str) -> None
        super(LoggedTrigger, self).__init__(name=name)
        self._logger = logger

    @property
    def logger(self):
        return self._logger


class Not(Trigger):
    """Triggers when the given trigger does not.

    Arguments:
        trigger (Trigger): trigger that should be negated
    """

    def __init__(self, trigger):
        # type: (trigger) -> None
        assert isinstance(trigger, Trigger)
        super(Not, self).__init__()
        self._trigger = trigger

    def __call__(self):
        return not self._trigger()


class MultinaryOperator(Trigger):
    """Base class for operators involving an arbitrary number of triggers.

    Arguments:
        triggers (Trigger): parameters of the operation.
    """

    def __init__(self, *triggers):
        # type: (Tuple[Trigger]) -> None
        assert all([isinstance(op, Trigger) for op in triggers])
        super(MultinaryOperator, self).__init__()
        self._triggers = triggers

    def __call__(self):
        raise NotImplemented


class Any(MultinaryOperator):
    """Returns `True` iff any of the given `triggers` returns `True`.

    Arguments:
        triggers (Trigger): parameters of the operation.
    """

    def __init__(self, *triggers):
        # type: (Tuple[Trigger]) -> None
        super(Any, self).__init__(*triggers)

    def __call__(self):
        return any([op() for op in self._triggers])


class All(MultinaryOperator):
    """Returns `True` iff all of the given `triggers` return `True`.

    Arguments:
        triggers (Trigger): parameters of the operation.
    """

    def __init__(self, *triggers):
        # type: (Tuple[Trigger]) -> None
        super(All, self).__init__(*triggers)

    def __call__(self):
        return all([op() for op in self._triggers])


class TriggerLogWrapper(object):
    def __init__(self, trigger, msg, *args, **kwargs):
        # type: (Trigger, str, Tuple[typing.Any], Dict[str, typing.Any]) -> None
        if trigger.name:
            self._msg = 'Trigger "' + trigger.name + '": ' + msg
        else:
            self._msg = msg

        self._args = args
        self._kwargs = kwargs

    def __str__(self):
        return self._msg.format(*self._args, **self._kwargs)