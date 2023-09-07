import numpy as np
import functools
import numbers
import sys

epsilon = sys.float_info.epsilon


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


def valid_butcher(A, b, c, s):

    valid_b = np.sum(b) == 1
    valid_c = np.sum(
        np.abs(np.array([np.sum(row) for row in A]) - c)) < 2*len(c)*epsilon
    valid_size = len(b) == s and A.shape[0] == s and A.shape[1] == s

    assert valid_b, "invalid b component of butcher array"
    assert valid_c, "invalid c component of butcher array"
    assert valid_size, "array sizes not compatible"