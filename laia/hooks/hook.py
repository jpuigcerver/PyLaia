from __future__ import absolute_import

from typing import Callable, Tuple


class Hook(object):
    """Executes an action when the condition is met"""

    def __init__(self, condition, action, *args, **kwargs):
        # type: (Callable, Callable) -> None
        assert condition is not None
        assert action is not None
        self._condition = condition
        self._action = action
        # Allow args and kwargs so that an action's
        # arguments can be specified inside the Hook
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        # Note that the hook's args come before call's args
        # and the hook's kwargs overwrite call's kwargs
        a = self._args + args
        kw = dict(kwargs, **self._kwargs)
        return self._action(*a, **kw) if self._condition() else False

    def state_dict(self):
        return {
            "condition": self._condition.state_dict()
            if hasattr(self._condition, "state_dict")
            else None,
            "action": self._action.state_dict()
            if hasattr(self._action, "state_dict")
            else None,
        }

    def load_state_dict(self, state):
        if hasattr(self._condition, "load_state_dict"):
            self._condition.load_state_dict(state["condition"])
        if hasattr(self._action, "load_state_dict"):
            self._action.load_state_dict(state["action"])


class HookCollection(object):
    r"""When called, calls a collection of :class:`~Hook`` objects."""

    def __init__(self, *hooks):
        # type: (Tuple[Callable]) -> None
        assert all(isinstance(h, Hook) for h in hooks)
        self._hooks = hooks

    def __call__(self, *args, **kwargs):
        return any([h(*args, **kwargs) for h in self._hooks])

    def state_dict(self):
        return {
            "hooks": [
                hook.state_dict() if hasattr(hook, "state_dict") else None
                for hook in self._hooks
            ]
        }

    def load_state_dict(self, state):
        for i, hook in enumerate(self._hooks):
            if i >= len(state["hooks"]):
                break
            if hasattr(hook, "load_state_dict"):
                hook.load_state_dict(state["hooks"][i])
