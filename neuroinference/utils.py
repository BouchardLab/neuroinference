"""General utility functions for the class."""
import h5py
import numpy as np
import matplotlib.pyplot as plt


def copy_group(send_file, send_group, rec_file, rec_group):
    """Copies one group in an h5py file, and pastes it in another file.

    Parameters
    ----------
    send_file : string
        The filepath containing the data to copy.

    send_group : string
        The path in the h5 file to copy.

    rec_file : string
        The filepath containing the h5 file to receive the data.

    rec_group : string
        The path in the h5 file to paste the data.
    """
    send = h5py.File(send_file, 'r')
    to_send = send[send_group]
    rec = h5py.File(rec_file, 'a')
    to_rec = rec.create_group(rec_group)

    for key, val in to_send.items():
        if isinstance(val, h5py.Dataset):
            if key != 'Y':
                to_rec[key.lower()] = val[:]
            else:
                to_rec[key] = val[:]
        else:
            group2 = to_rec.create_group(key)
            for key2, val2 in val.items():
                group2[key2] = val2[:]

    send.close()
    rec.close()


def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cohens_d(first, second):
    """Calculates Cohen's d between two samples."""
    # sample sizes
    n1 = first.size
    n2 = second.size
    # means
    mu1 = np.mean(first)
    mu2 = np.mean(second)
    # standard deviations
    s1 = np.std(first)
    s2 = np.std(second)
    # pool standard deviations
    s = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    # cohen's d
    d = (mu1 - mu2) / s
    return d


def tighten_scatter_plot(ax, bounds, line_kwargs=None):
    """Takes an axis and makes the x and y limits equal, while plotting an
    identity line.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to tighten.

    bounds : tuple or array-like
        The bounds of the x and y axes.

    line_color : string
        The color of the identity line.
    """
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    if line_kwargs is not None:
        ax.plot(bounds, bounds,
                color=line_kwargs.get('color', 'gray'),
                linewidth=line_kwargs.get('linewidth', 3),
                linestyle=line_kwargs.get('linestyle', '-'),
                zorder=line_kwargs.get('zorder', -1))
    ax.set_aspect('equal')
    return ax


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


def deviance_poisson(y_true, y_pred):
    """Calculates the deviance for a Poisson model.

    Parameters
    ----------
    y_true : ndarray
        The true response values.

    y_pred : ndarray
        The predicted response values, according to the Poisson model.

    Returns
    -------
    dev : float
        The total deviance of the data.
    """
    # calculate log-likelihood of the predicted values
    ll_pred = np.sum(y_true * np.log(y_pred) - y_pred)
    # calculate log-likelihood of the true data
    y_true_nz = y_true[y_true != 0]
    ll_true = np.sum(y_true_nz * np.log(y_true_nz) - y_true_nz)
    # calculate deviance
    deviance = ll_true - ll_pred
    return deviance


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


def plot_metric_summary(
    baseline_group, fits_groups, metrics, fax=None, omit_idxs=None
):
    """Analyze a set of model fits in a grid of subplots, using a user-provided
    set of metrics.

    Parameters
    ----------
    baseline_group : HDF5 Object
        The baseline algorithm to compare against.

    fits_groups : list of HDF5 objects
        A list of the fits to look at.

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

        n_total_units = baseline_selection_ratio.size
        selected_idxs = np.setdiff1d(np.arange(n_total_units), omit_idxs)

        # iterate over algorithms
        for col_idx, algorithm in enumerate(fits_groups):
            if metric == 'selection_ratio':
                # calculate selection ratio for algorithm
                coefs = algorithm[key][:]
                selection_ratio = calculate_selection_ratio(coefs).mean(axis=0)

                # plot direct comparison
                axes[row_idx, col_idx].scatter(
                    baseline_selection_ratio[selected_idxs],
                    selection_ratio[selected_idxs],
                    alpha=0.5,
                    color='k',
                    edgecolor='w')
            else:
                # plot some metric already stored in the H5 file
                axes[row_idx, col_idx].scatter(
                    baseline_group[metric][:].mean(axis=0)[selected_idxs],
                    algorithm[metric][:].mean(axis=0)[selected_idxs],
                    alpha=0.5,
                    color='k',
                    edgecolor='w')

    return fig, axes


def plot_difference_distribution(
    baseline_group, fits_groups, metrics, fax=None, alpha=0.5, color='gray'
):
    """Assess the difference between two metrics for a baseline and a comparison
    set of fits.

    Parameters
    ----------
    baseline_group : HDF5 Object
        The baseline algorithm to compare against.

    fits_groups : list of HDF5 objects
        A list of the fits to look at.

    metrics : list of strings
        A list of the metrics to plot the differences for.

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
                                 figsize=(4 * n_algorithms, 2.5 * n_metrics))
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
                axes[row_idx, col_idx].hist(
                    selection_ratio - baseline_selection_ratio,
                    alpha=alpha,
                    color=color)
            else:
                # plot some metric already stored in the H5 file
                axes[row_idx, col_idx].hist(
                    (baseline_group[metric][:].mean(axis=0)
                     - algorithm[metric][:].mean(axis=0)),
                    alpha=alpha,
                    color=color)

    return fig, axes


def append_cb_ax(fig, ax, width=0.015, x_adj=0, y_adj=0):
    """Appends a colorbar axis to a panel of subplots.

    Parameters
    ----------
    fig, ax : matplotlib objects
        Figure and axes for a matplotlib figure.

    width : float
        The width of the figure.

    x_adj : float
        The amount to horizontally displace the colorbar.

    y_adj : float
        The amount to vertically displace the colorbar.

    Returns
    -------
    cax : colorbar axes
        The colorbar axes.
    """
    fig_invert = fig.transFigure.inverted()
    ax_bottom_right_x, ax_bottom_right_y = \
        fig_invert.transform(ax.transAxes.transform([1.0, 0.]))
    ax_top_right_x, ax_top_right_y = \
        fig_invert.transform(ax.transAxes.transform([1.0, 1.0]))

    cax = fig.add_axes([
        ax_bottom_right_x + x_adj,
        ax_bottom_right_y + y_adj,
        width,
        ax_top_right_y - ax_bottom_right_y])

    return cax


def selection_accuracy(true, est):
    """Calculate the selection accuracy from an estimated profile, given
    the true selection profile.

    Parameters
    ----------
    true, est : np.ndarray
        The true and fitted parameters.

    Returns
    -------
    selection_accuracy : int
        The selection accuracy.
    """
    true = true.astype(bool)
    est = est.astype(bool)
    S_true = np.count_nonzero(true)
    S_est = np.count_nonzero(est)
    S_diff = np.count_nonzero(np.logical_xor(true, est))
    return 1 - (S_diff / (S_true + S_est))
