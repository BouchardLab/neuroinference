import matplotlib.pyplot as plt
import numpy as np

from .utils import idx_to_xy


def plot_border(x, y, which, ax, linewidth=2):
    """Plots the border for a cell in an imshow plot.

    Parameters
    ----------
    x : int
        The x-coordinate of the cell.

    y : int
        The y-coordinate of the cell.

    which : string
        Which side of the cell to place a border.

    ax : matplotlib object
        The axis on which to place the border.

    linewidth : float
        The width of the border.

    Returns
    -------
    ax : matplotlib object
        The axis on which to place the border.
    """
    if which == 'left':
        xs = (x - 0.5) * np.ones(1000)
        ys = np.linspace(y - 0.5, y + 0.5, 1000)
    elif which == 'right':
        xs = (x + 0.5) * np.ones(1000)
        ys = np.linspace(y - 0.5, y + 0.5, 1000)
    elif which == 'top':
        xs = np.linspace(x - 0.5, x + 0.5, 1000)
        ys = (y + 0.5) * np.ones(1000)
    elif which == 'bottom':
        xs = np.linspace(x - 0.5, x + 0.5, 1000)
        ys = (y - 0.5) * np.ones(1000)
    ax.plot(xs, ys, color='k', linewidth=linewidth)
    return ax


def plot_basal_ganglia_coefs(
    baseline_coefs, uoi_coefs, vmin=-1.75, vmax=1.75, row_columns=(3, 6),
    scattersize=225, fax=None
):
    """Plot the basal ganglia coefficients in a grid.

    Parameters
    ----------
    baseline_coefs : np.ndarray, shape (n_folds, n_neurons)
        The baseline coefficients.

    uoi_coefs : np.ndarray, shape (n_folds, n_neurons)
        The UoI coefficients.

    vmin : float
        The minimum value of the coefficient to include on the imshow.

    vmax : float
        The maximum value of the coefficient to include on the imshow.

    row_columns : tuple

    scattersize : int
        The size of an 'x' to show for coefficients that are zero.

    fax : matplotlib figure and axes
        The figure and axes to plot on.

    Returns
    -------
    fig, axes : matplotlib figure and axes
        The figure and axes containing the plot.

    baseline_img, uoi_img : imshow objects
        The imshow objects for the baseline and UoI coefficients.
    """
    # create figure if not provided
    if fax is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    else:
        fig, axes = fax

    # calculate median coefficient values
    baseline_coefs = np.median(baseline_coefs, axis=0)
    uoi_coefs = np.median(uoi_coefs, axis=0)

    # plot baseline coefficients
    baseline_img = axes[0].imshow(
        np.flip(baseline_coefs.reshape(row_columns), axis=0),
        vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap('RdGy'))

    # for zero coefficients, show an x
    for idx, coef in enumerate(baseline_coefs):
        if coef == 0:
            x, y = idx_to_xy(idx, row_columns[1], row_columns[0])
            axes[0].scatter(x, y, marker='x', color='k', s=scattersize)

    # plot uoi coefficients
    uoi_img = axes[1].imshow(
        np.flip(uoi_coefs.reshape(row_columns), axis=0),
        vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap('RdGy'))

    # for zero coefficients, show an x
    for idx, coef in enumerate(uoi_coefs):
        if coef == 0:
            x, y = idx_to_xy(idx, row_columns[1], row_columns[0])
            axes[1].scatter(x, y, marker='x', color='k', s=scattersize)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axes, baseline_img, uoi_img
