def _copy_list(l, dispatch):
    ret = l[:]
    for idx, item in enumerate(ret):
        cp = dispatch.get(type(item))
        if cp is not None:
            ret[idx] = cp(item, dispatch)
    return ret


def _copy_dict(d, dispatch):
    ret = d.copy()
    for key, value in ret.items():
        cp = dispatch.get(type(value))
        if cp is not None:
            ret[key] = cp(value, dispatch)
    return ret


def _copy_set(d ,dispatch):
    ret = d.copy()
    return ret


_dispatcher = {
    list: _copy_list,
    dict: _copy_dict,
    set: _copy_set,
}


def deepcopy(sth):
    cp = _dispatcher.get(type(sth))
    if cp is None:
        return sth
    else:
        return cp(sth, _dispatcher)
