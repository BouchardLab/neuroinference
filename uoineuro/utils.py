import numpy as np
import matplotlib.pyplot as plt


def idx_to_xy(idx, dx, dy):
    """Converts an index to a set of xy coordinates in a grid with specified
    width and height.

    Parameters
    ----------
    idx : int
        The index.

    dx : int
        The length of the grid.

    dy : int
        The width of the grid.

    Returns
    -------
    x, y : int
        The x and y coordinates.
    """
    x = idx % dx
    y = dy - 1 - (idx // dx)
    return x, y


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


def plot_metric_summary(baseline_group, fits_groups, metrics, fax=None):
    """Analyze a set of coupling fits in a grid of subplots, using a user-provided
    set of metrics.

    Parameters
    ----------
    baseline_group : HDF5 Object
        The baseline algorithm to compare against.

    fits_groups : list of HDF5 objects
        A list of the coupling fits to look at.

    metrics : list of strings
        A list of the metrics to plot in the rows of the subplots.

    fax : tuple of (fig, axes) matplotlib objects
        If None, a (fig, axes) is created. Otherwise, fax are modified directly.

    Returns
    -------
    fax : tuple of (fig, axes) matplotlib objects
        The (fig, axes) on which the metrics were plotted.
    """
    n_algorithms = len(fits_groups)
    n_metrics = len(metrics)

    if fax is None:
        fig, axes = plt.subplots(n_metrics, n_algorithms,
                                 figsize=(3 * n_algorithms, 3 * n_metrics))
    else:
        fig, axes = fax

    # iterate over metrics
    for row_idx, metric in enumerate(metrics):
        if metric == 'selection_ratio':
            # extract the key containing the coefficient dataset
            key = [key for key in baseline_group if 'coefs' in key][0]
            baseline_coefs = baseline_group[key][:]
            # calculate selection ratio for baseline
            baseline_selection_ratio = \
                calculate_selection_ratio(baseline_coefs).mean(axis=0)

        # iterate over algorithms
        for col_idx, algorithm in enumerate(fits_groups):
            if metric == 'selection_ratio':
                # calculate selection ratio for algorithm
                coefs = algorithm[key][:]
                selection_ratio = calculate_selection_ratio(coefs).mean(axis=0)

                # plot direct comparison
                axes[row_idx, col_idx].scatter(
                    baseline_selection_ratio,
                    selection_ratio,
                    alpha=0.5,
                    color='k',
                    edgecolor='w')
            else:
                # plot some metric already stored in the H5 file
                axes[row_idx, col_idx].scatter(
                    baseline_group[metric][:].mean(axis=0),
                    algorithm[metric][:].mean(axis=0),
                    alpha=0.5,
                    color='k',
                    edgecolor='w')

    return fig, axes
