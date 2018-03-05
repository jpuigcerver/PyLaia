from __future__ import absolute_import

from laia.engine.triggers.trigger import Trigger


class MultinaryOperator(Trigger):
    r"""Base class for operators involving multiple triggers."""
    def __init__(self, *triggers):
        assert all([isinstance(op, Trigger) for op in triggers])
        self._triggers = triggers

    def __call__(self, trainer):
        raise NotImplemented


class Any(MultinaryOperator):
    r"""Returns `True` iff any of the given `triggers` returns `True`."""
    def __init__(self, *triggers):
        super(Any, self).__init__(*triggers)

    def __call__(self):
        return any([op() for op in self._triggers])


class All(MultinaryOperator):
    r"""Returns `True` iff all of the given `triggers` return `True`."""
    def __init__(self, *triggers):
        super(All, self).__init__(*triggers)

    def __call__(self):
        return all([op() for op in self._triggers])


class Not(Trigger):
    r"""Negates the value of the given trigger."""
    def __init__(self, trigger):
        super(Not, self).__init__()
        assert isinstance(trigger, Trigger)
        self._trigger = trigger

    def __call__(self):
        return not self._trigger()
