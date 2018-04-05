from __future__ import absolute_import

import inspect
from functools import wraps
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


class HookCollection(object):
    r"""When called, calls a collection of :class:`~Hook`` objects."""

    def __init__(self, *hooks):
        # type: (Tuple[Callable]) -> None
        assert all(isinstance(h, Hook) for h in hooks)
        self._hooks = hooks

    def __call__(self, *args, **kwargs):
        return any(h(*args, **kwargs) for h in self._hooks)


def action(func):
    """Decorator.

    Filters the number of arguments passed using
    the number of non kwarg arguments in the signature.

    Also filters all kwargs passed which
    are not part of the function parameters

    Note: It does not make sense to take `*args, **kwargs`
    in an `@action` annotated function since any possible
    value passed will be filtered by the wrapper.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            argspec = inspect.getfullargspec(func)
        except AttributeError:
            argspec = inspect.getargspec(func)

        non_kwargs_num = len(argspec.args) - len(argspec.defaults or [])
        return func(*args[:non_kwargs_num],
                    **{k: v for k, v in kwargs.items()
                       if k in argspec.args})

    return wrapper
