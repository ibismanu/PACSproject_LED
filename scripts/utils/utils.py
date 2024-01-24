import numpy as np
import functools
from scipy import ndimage


def to_numpy(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        ret = fun(*args, **kwargs)
        if np.isscalar(ret):
            ret = np.array([ret])
        if not isinstance(ret, np.ndarray):
            ret = np.array(ret)
        return ret

    return wrapper


def check_butcher_sum(A, b, c, s):
    tol = 1e-5
    valid_b = np.sum(b) == 1
    valid_c = np.sum(np.abs(np.array([np.sum(row) for row in A]) - c)) < tol
    valid_size = len(b) == s and A.shape[0] == s and A.shape[1] == s

    assert valid_b, "invalid b component of butcher array"
    assert valid_c, "invalid c component of butcher array"
    assert valid_size, "array sizes not compatible"


def check_explicit_array(A, semi=False):
    if semi:
        return np.array_equal(A, np.tril(A))
    else:
        return np.array_equal(A, np.tril(A, -1))


def integral(g, j, p):
    deg = (p + 1) // 2
    nodes, weights = np.polynomial.legendre.leggauss(deg=deg)
    nodes = 0.5 * (nodes + 1)
    weights = 0.5 * weights
    return np.sum(np.array([weights[i] * g(nodes[i], j) for i in range(deg)]))


def build_sequences(data, window, stride=1, telescope=1):
    # data should have shape (latent_dim, timesteps)

    assert window % stride == 0

    data = np.transpose(data)

    dataset = []
    target = []

    for idx in np.arange(0, data.shape[1] - window - telescope, stride):
        dataset.append(np.transpose(data[:, idx : idx + window]))
        target.append(np.transpose(data[:, idx + window : idx + window + telescope]))

    return np.array(dataset), np.array(target)


def import_tensorflow():
    # Filter tensorflow version warnings
    import os

    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    import warnings

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)
    import logging

    tf.get_logger().setLevel(logging.ERROR)
    return tf


def smooth_filter(y, window_size, order):
    half_window = (window_size - 1) // 2

    b = np.mat(
        [
            [k**i for i in range(order + 1)]
            for k in range(-half_window, half_window + 1)
        ]
    )

    m = np.linalg.pinv(b).A[0]

    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode="valid")
