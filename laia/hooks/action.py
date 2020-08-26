from functools import wraps
from inspect import getfullargspec
from typing import Any, Callable


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
    def wrapper(*args: Any, **kwargs: Any):
        argspec = getfullargspec(func)
        non_kwargs = len(argspec.args) - len(argspec.defaults or [])
        return func(
            *args[:non_kwargs], **{k: v for k, v in kwargs.items() if k in argspec.args}
        )

    return wrapper


class Action:
    def __init__(self, callable_: Callable, *args: Any, **kwargs: Any) -> None:
        self._callable = callable_
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any):
        a = self._args + args
        kw = {**self._kwargs, **kwargs}
        return self._callable(*a, **kw)


class ActionList:
    """When called, calls a collection of :class:`~Action` objects."""

    def __init__(self, *actions: Callable) -> None:
        assert all(isinstance(a, Action) for a in actions)
        self._actions = actions

    def __call__(self, *args: Any, **kwargs: Any):
        for action in self._actions:
            action(*args, **kwargs)
