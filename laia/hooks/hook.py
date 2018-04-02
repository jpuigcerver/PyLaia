from __future__ import absolute_import

from functools import wraps
from typing import Callable, Tuple


class Hook(object):
    """Executes an action when the condition is met"""

    def __init__(self, condition, action):
        # type: (Callable, Callable) -> None
        assert condition is not None
        assert action is not None
        self._condition = condition
        self._action = action

    def __call__(self, *args, **kwargs):
        return self._action(*args, **kwargs) if self._condition() else False


class HookCollection(object):
    r"""When called, calls a collection of :class:`~Hook`` objects."""

    def __init__(self, *hooks):
        # type: (Tuple[Callable]) -> None
        assert all(isinstance(h, Hook) for h in hooks)
        self._hooks = hooks

    def __call__(self, *args, **kwargs):
        return any(h(*args, **kwargs) for h in self._hooks)


def action_kwargs(*keys):
    """`keys` is used to filter the kwargs passed to the function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **{
                k: v for k, v in kwargs.items() if k in keys})

        return wrapper

    return decorator
