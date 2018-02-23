from __future__ import absolute_import

from laia.engine.savers.saver import Saver


class TriggerSaver(object):
    def __init__(self, trigger, saver):
        assert trigger is not None
        assert saver is not None
        self._trigger = trigger
        self._saver = saver

    def __call__(self, obj):
        if self._trigger():
            return saver(obj)
        else:
            return False
