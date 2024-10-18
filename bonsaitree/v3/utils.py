from typing import Any


def get_pairs(
    lst: list[Any],
):
    """
    For a list of the form [l1, l2, l3, ..., lk]
    return [(l1,l2), (l2,l3), ..., (lk-1,lk)]

    Args:
        lst: list of the form [l1, l2, l3, ..., lk]

    Returns:
        pairs: list of the form
                [(l1,l2), (l2,l3), ..., (lk-1,lk)]
    """
    n = len(lst)
    # initialize counter
    ctr = 0
    # loop until counter is less than n
    while ctr < n-1:
        # produce the current ctr of the counter
        yield (lst[ctr], lst[ctr+1])
        # increment the counter
        ctr += 1


def unstringify_keys(
    str_dct: dict[str, Any],
):
    """
    Unstringify the keys of a dictionary.

    Args:
        str_dct (dict): A possibly nested dictionary with keys
            expressed as strings. Subdicts can also have
            string keys.

    Returns:
        dct: str_dict with keys unstringified.
    """
    def try_eval(s):
        try:
            return eval(s)
        except NameError:
            if s == 'Infinity':
                return float('inf')
            elif s == 'NaN':
                return None
            else:
                return s

    if type(str_dct) is not dict:
        return str_dct
    else:
        dct = {}
        for k, v in str_dct.items():
            new_k = try_eval(k)
            dct[new_k] = unstringify_keys(v)
    return dct
