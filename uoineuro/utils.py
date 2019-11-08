import numpy as np


def calculate_selection_ratio(coefs):
    """Calculate the selection ratio, or fraction of non-zero parameters, for a
    set of coefficients.

    Parameters
    ----------
    coefs : nd-array
        A vector (or matrix) of coefficient values. The last axis is assumed
        to contain the number of parameters.

    Returns
    -------
    selection_ratio : float or nd-array
        The fraction of non-zero parameter values. If coefs is a vector, then
        selection_ratio is a float. If coefs is an nd-array, selection_ratio
        is also an nd-array with one fewer dimension.
    """
    n_max_coefs = coefs.shape[-1]
    selection_ratio = np.count_nonzero(coefs, axis=-1) / n_max_coefs
    return selection_ratio
