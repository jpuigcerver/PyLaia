import inspect
from functools import wraps


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
