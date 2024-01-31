import numpy as np
import numba as nb


@nb.jit(nopython=True)
def gaussian(x, mu=0.0, sig=1.0):
    return (np.exp(-0.5 * np.square((x - mu) / sig))
            / (sig * np.sqrt(2.0 * np.pi)))


@nb.jit(nopython=True)
def rectangular(x, center=0.0, halfwidth=0.5):
    return np.logical_and(
        np.greater_equal(x - center, -halfwidth),
        np.less(x - center, halfwidth))


@nb.jit(nopython=True)
def triangular(x, center=0.0, halfwidth=0.5):
    slope = -2 / halfwidth
    dom_f = (a := x - center)[abs(a) <= halfwidth]
    return 2 + (slope * abs(dom_f))
