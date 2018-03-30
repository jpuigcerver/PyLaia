from __future__ import absolute_import


class Hook(object):
    """Executes an action when the condition is met"""

    def __init__(self, condition, action):
        assert condition is not None
        assert action is not None
        self._condition = condition
        self._action = action

    def __call__(self, **kwargs):
        if self._condition():
            return self._action(**kwargs)
        else:
            return False


class HookCollection(object):
    r"""When called, calls a collection of :class:`~Hook`` objects."""

    def __init__(self, *hooks):
        assert all(isinstance(h, Hook) for h in hooks)
        self._hooks = hooks

    def __call__(self, *args, **kwargs):
        return any(h(*args, **kwargs) for h in self._hooks)
