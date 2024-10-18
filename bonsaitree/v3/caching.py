import functools
from typing import Any, Callable

from frozendict import frozendict


def freeze_dict(
    dct: dict[Any, Any],
):
    """
    Recursively freeze a nested dictionary. Useful for
    creating hashable input to memoized functions that
    take unhashable dicts as arguments.
    """

    if type(dct) is dict or type(dct) is frozendict:
        return frozendict( {k : freeze_dict(v) for k,v in dct.items()} )
    elif isinstance(dct, list):
        return tuple([freeze_dict(e) for e in dct])
    elif isinstance(dct, set):
        return frozenset({freeze_dict(e) for e in dct})
    else:
        try:
            return dct
        except TypeError: # not hashable
            if hasattr(dct, '__iter__'):
                return frozenset([freeze_dict(v) for v in dct])


def freeze_args(
    func: Callable[..., Any],
):
    """
    Make a decorator to freeze the arguments to a function
    """
    @functools.wraps(func)
    def wrapped(*args,**kwargs):
        args = tuple([freeze_dict(a) for a in args])
        kwargs = {k : freeze_dict(v) for k,v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped


def unfreeze_dict(
    dct: dict[Any, Any],
):
    """
    Recursively unfreeze a nested dictionary. Useful for
    creating hashable input to memoized functions that
    take unhashable dicts as arguments.
    """

    if type(dct) == frozendict:
        return {k : unfreeze_dict(v) for k,v in dct.items()}
    else:
        return dct
