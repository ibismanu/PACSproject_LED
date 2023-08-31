import numpy as np
import functools
import numbers


def to_numpy(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        ret = fun(*args, **kwargs)
        if isinstance(ret, numbers.Number):
            ret = np.array([ret])
        if not isinstance(ret, np.ndarray):
            ret = np.array(ret)
        return ret
    return wrapper
