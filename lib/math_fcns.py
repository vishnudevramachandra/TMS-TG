import numpy as np
import numba as nb


@nb.jit(nopython=True)
def gaussian(x, mu=0.0, sig=1.0):
    return (np.exp(-0.5 * np.square((x - mu) / sig))
            / (sig * np.sqrt(2.0 * np.pi)))


@nb.jit(nopython=True)
def rectangular(x, center=0.0, width=1.0):
    return np.logical_and(
        np.greater_equal(x - center, -width / 2),
        np.less(x - center, width / 2))


@nb.jit(nopython=True)
def triangular(x, center=0.0, width=1.0):
    slope = -2 / (width / 2)
    dom_f = (a := x - center)[abs(a) <= (width / 2)]
    return 2 + (slope * abs(dom_f))
