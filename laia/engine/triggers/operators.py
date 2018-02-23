from __future__ import absolute_import

from laia.engine.triggers.trigger import Trigger


class Not(Trigger):
    def __init__(self, trigger):
        super(Not, self).__init__()
        assert isinstance(trigger, Trigger)
        self._trigger = trigger

    def __call__(self):
        return not self._trigger()


class Operator(Trigger):
    def __init__(self, *triggers):
        assert all([isinstance(op, Trigger) for op in triggers])
        self._triggers = triggers

    def __call__(self, trainer):
        raise NotImplemented


class Any(Operator):
    def __init__(self, *triggers):
        super(Any, self).__init__(*triggers)

    def __call__(self):
        return any([op() for op in self._triggers])


class All(Operator):
    def __init__(self, *triggers):
        super(All, self).__init__(*triggers)

    def __call__(self):
        return all([op() for op in self._triggers])
