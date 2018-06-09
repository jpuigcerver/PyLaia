import inspect
from functools import wraps

from typing import Callable, Any, Tuple


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
        return func(
            *args[:non_kwargs_num],
            **{k: v for k, v in kwargs.items() if k in argspec.args}
        )

    return wrapper


class Action(object):
    def __init__(self, callable_, *args, **kwargs):
        # type: (Callable, Any, Any) -> None
        self._callable = callable_
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        a = self._args + args
        kw = dict(self._kwargs, **kwargs)
        return self._callable(*a, **kw)


class ActionCollection(object):
    """When called, calls a collection of :class:`~Action` objects."""

    def __init__(self, *actions):
        # type: (Tuple[Callable]) -> None
        assert all(isinstance(a, Action) for a in actions)
        self._actions = actions

    def __call__(self, *args, **kwargs):
        for action in self._actions:
            action(*args, **kwargs)
