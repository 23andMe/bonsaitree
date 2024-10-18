import ast
import functools
import json
from typing import Any

from .constants import (
    AGE_MOMENT_FP,
    IBD_MOMENT_FP,
)
from .utils import unstringify_keys


def eval_str(
    obj: Any,
):
    """
    Evaluate an object to a literal.

    Args:
        obj: an object.

    Returns:
        e_obj: evaluated version of object.
            if obj is an evaluatable string,
            returns the literal value of obj.
            Otherwise, returns obj.
    """
    try:
        return ast.literal_eval(obj)
    except ValueError:
        return obj


@functools.lru_cache(maxsize=None)
def load_age_moments():
    str_age_moments = json.loads(open(AGE_MOMENT_FP, 'r').read())
    return unstringify_keys(str_age_moments)


@functools.lru_cache(maxsize=None)
def load_ibd_moments():
    str_ibd_moments = json.loads(open(IBD_MOMENT_FP, 'r').read())
    return unstringify_keys(str_ibd_moments)
