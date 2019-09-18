# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: helpers.py
# time: 2:47 下午
import inspect
import warnings
from functools import wraps

string_types = (type(b''), type(u''))


def deprecated(reason):
    """Decorator to mark functions as deprecated.

    Calling a decorated function will result in a warning being emitted, using warnings.warn.
    Adapted from https://stackoverflow.com/a/40301488/8001386.

    Parameters
    ----------
    reason : str
        Reason of deprecation.

    Returns
    -------
    function
        Decorated function

    """
    if isinstance(reason, string_types):
        def decorator(func):
            @wraps(func)
            def new_func1(*args, **kwargs):
                warnings.warn(
                    f"Call to deprecated `{func.__name__}` ({reason}).",
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func = reason

        @wraps(func)
        def new_func2(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated `{func.__name__}`.",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


if __name__ == "__main__":
    print("hello, world")
