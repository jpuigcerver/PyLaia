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
        kw = {**kwargs, **self._kwargs}
        return self._action(*a, **kw) if self._condition() else False

    def state_dict(self):
        return {
            k: v.state_dict() if hasattr(v, "state_dict") else None
            for k, v in (("condition", self._condition), ("action", self._action))
        }

    def load_state_dict(self, state):
        for k, v in ("condition", self._condition), ("action", self._action):
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(state[k])


class HookList(object):
    """When called, calls a collection of :class:`~Hook`` objects."""

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
        state = state["hooks"]
        for i, hook in enumerate(self._hooks):
            if i >= len(state):
                break
            if hasattr(hook, "load_state_dict"):
                hook.load_state_dict(state[i])
