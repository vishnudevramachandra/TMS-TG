import numpy as np
import numba as nb


@nb.jit(nopython=True)
def gaussian(x, mu=0.0, sig=1.0):
    """
    Compute the value of the Gaussian function for the given input.

    Parameters:
    - x: Input value or array
    - mu: Mean of the Gaussian distribution (default is 0.0)
    - sig: Standard deviation of the Gaussian distribution (default is 1.0)

    Returns:
    - Value of the Gaussian function at the given input(s)
    """

    return (np.exp(-0.5 * np.square((x - mu) / sig))
            / (sig * np.sqrt(2.0 * np.pi)))


@nb.jit(nopython=True)
def rectangular(x, center=0.0, halfwidth=0.5):
    """
    Compute the value of the rectangular function for the given input.

    Parameters:
    - x: Input value or array
    - center: Center of the rectangular function (default is 0.0)
    - halfwidth: Half-width of the rectangular function (default is 0.5)

    Returns:
    - Boolean array indicating whether the input value(s) falls within the rectangular window
    """

    return np.logical_and(
        np.greater_equal(x - center, -halfwidth),
        np.less(x - center, halfwidth))


@nb.jit(nopython=True)
def triangular(x, center=0.0, halfwidth=0.5):
    """
    Compute the value of the triangular function for the given input.

    Parameters:
    - x: Input value or array
    - center: Center of the triangular function (default is 0.0)
    - halfwidth: Half-width of the triangular function (default is 0.5)

    Returns:
    - Value of the triangular function at the given input(s)
    """

    slope = -2 / halfwidth
    dom_f = (a := x - center)[abs(a) <= halfwidth]      # Compute the domain of the function where the value is non-zero

    return 2 + (slope * abs(dom_f))
