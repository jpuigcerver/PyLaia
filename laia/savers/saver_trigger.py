from __future__ import absolute_import

from laia.savers.saver import Saver


class SaverTrigger(object):
    r"""When called, saves the given object the ``saver`` if the ``trigger``
    returns True.
    """
    def __init__(self, trigger, saver):
        assert trigger is not None
        assert saver is not None
        self._trigger = trigger
        self._saver = saver

    def __call__(self, obj):
        if self._trigger():
            return self._saver(obj)
        else:
            return False


class SaverTriggerCollection(object):
    r"""When called, calls a collection of :class:`~SaverTrigger`` objects."""
    def __init__(self, *saver_triggers):
        assert all([isinstance(st, SaverTrigger) for st in saver_triggers])
        self._saver_triggers = saver_triggers

    def __call__(self, obj):
        return any([st(obj) for st in self._saver_triggers])
