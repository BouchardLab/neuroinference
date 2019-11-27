import numpy as np

from pykalman import KalmanFilter
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import r2_score
from sklearn.utils import check_random_state


def stratify_sample(stratify, frac, random_state=None):
    """Samples proportionally according to class labels.

    Parameters
    ----------
    stratify : array-like, shape (n_samples,)
        The set of class labels for each sample.

    frac : float
        The fraction of entries to subsample.

    random_state : RandomState() or None
        A random state for subsampling.

    Returns
    -------
    selected_idx : np.ndarray
        A sorted list of sampled indices.
    """
    random_state = check_random_state(random_state)

    # get unique class labels
    groups = np.unique(stratify)
    selected_idx = np.array([], dtype='int')

    # iterate over class labels, choosing indices for each one
    for group in groups:
        # get samples in each group
        group_idx = np.argwhere(stratify == group).ravel().astype('int')
        n_selected = int(frac * group_idx.size)
        selected = random_state.choice(group_idx,
                                       size=n_selected,
                                       replace=False)
        selected_idx = np.append(selected_idx, selected)

    return np.sort(selected_idx)


def bi_cross_validator(
    X, nmf, ranks, row_frac=0.75, col_frac=0.75, n_reps=10, stratify=None,
    random_state=None
):
    """Performs Bi-Cross-Validation to choose an optimal rank for a
    non-negative matrix factorization.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data matrix.

    nmf : object
        Non-negative matrix factorization solver. Must come equipped with a
        fit() function.

    ranks : array-like
        A list of ranks from which to determine the optimal number of bases.

    row_frac : float
        The fraction of rows to use in the cross-validation.

    col_frac : float
        The fraction of columns to use in the cross-validation.

    n_reps : int
        Number of repetitions to run.

    stratify : array-like, shape (n_samples,)
        A list of class labels to stratify the subsampling.

    random_state : RandomState() or None
        A random state for seeding the subsampling.

    Returns
    -------
    k_max : int
        The rank that minimizes validation loss.

    errors : np.ndarray, shape (n_ranks,)
        An array containing the validation errors for each rank.
    """
    rng = check_random_state(random_state)

    n_samples, n_features = X.shape
    errors = np.zeros(ranks.size)

    # if no stratification, then every sample is in the same class
    if stratify is None:
        stratify = np.ones(n_samples)

    # iterate over repetitions of calculating validation loss
    for rep in range(n_reps):
        # iterate over the ranks
        for idx, k in enumerate(ranks):
            # choose rows according to stratification
            rows = stratify_sample(stratify, row_frac, rng=random_state)
            # choose columns randomly
            cols = np.sort(rng.choice(n_features,
                                      size=int(col_frac * n_features),
                                      replace=False))

            # decompose bottom right quadrant
            X_negI_negJ = np.delete(np.delete(X, rows, axis=0), cols, axis=1)
            nmf.set_params(n_components=k)
            nmf.fit(X_negI_negJ)
            H_negI_negJ = nmf.components_
            W_negI_negJ = nmf.transform(X_negI_negJ)

            # side blocks
            X_I_negJ = np.delete(X, cols, axis=1)[rows]
            X_negI_J = np.delete(X, rows, axis=0)[:, cols]

            # fit coefficients in last block
            W_IJ = np.zeros((rows.size, k))
            H_IJ = np.zeros((k, cols.size))

            for row in range(W_IJ.shape[0]):
                W_IJ[row] = nnls(H_negI_negJ.T, X_I_negJ[row])[0]

            for col in range(H_IJ.shape[1]):
                H_IJ[:, col] = nnls(W_negI_negJ, X_negI_J[:, col])[0]

            # calculate error on reconstruction
            X_IJ = X[rows][:, cols]
            X_IJ_hat = np.dot(W_IJ, H_IJ)
            errors[idx] += np.sum((X_IJ - X_IJ_hat)**2)
    # determine optimal rank
    k_max = ranks[np.argmin(errors)]
    return k_max, errors


def apply_linear_decoder(x, y, Y, train_frac=0.8):
    """Trains a linear decoder on incoming neural data, and applies it to test
    data.

    Parameters
    ----------
    x : np.ndarray
        Array containing the x positions.

    y : np.ndarray
        Array containing the y positions.

    Y : np.ndarray
        Neural activity array.

    train_frac : float
        The fraction of data to train on.

    Returns
    -------
    X_test : np.ndarray
        True positions/velocities.

    X_test_hat : np.ndarray
        Estimated positions/velocities.

    r2s : np.ndarray
        Coefficient of determination on test set.

    corrs : np.ndarray
        Correlations between estimated and true sets.
    """
    n_total_samples = Y.shape[0]
    n_train_samples = int(n_total_samples * train_frac)

    X = np.vstack((x, y)).T
    X_train = X[:n_train_samples]
    X_test = X[n_train_samples:-1]

    Z_train = Y[:n_train_samples]
    Z_test = Y[n_train_samples:-1]

    ols = LinearRegression(fit_intercept=True)
    ols.fit(Z_train, X_train)
    X_test_hat = ols.predict(Z_test)

    n_outputs = X_test.shape[1]
    r2s = np.zeros(n_outputs)
    corrs = np.zeros(n_outputs)
    # coefficient of determination
    for idx in range(n_outputs):
        r2s[idx] = r2_score(X_test[:, idx], X_test_hat[:, idx])
    # correlation
    for idx in range(n_outputs):
        corrs[idx] = np.corrcoef(X_test[:, idx], X_test_hat[:, idx])[0, 1]

    return X_test, X_test_hat, r2s, corrs


def apply_kalman_filter(x, y, Y, dt=0.25, train_frac=0.8):
    """Trains a Kalman Filter to incoming neural data, and applies it to test
    data.

    Parameters
    ----------
    x : np.ndarray
        Array containing the x positions.

    y : np.ndarray
        Array containing the y positions.

    Y : np.ndarray
        Neural activity array.

    dt : float
        Bin width.

    train_frac : float
        The fraction of data to train on.

    Returns
    -------
    X_test : np.ndarray
        True positions/velocities.

    X_test_hat : np.ndarray
        Estimated positions/velocities.

    r2s : np.ndarray
        Coefficient of determination on test set.

    corrs : np.ndarray
        Correlations between estimated and true sets.
    """
    n_total_samples = Y.shape[0]
    n_train_samples = int(n_total_samples * train_frac)

    # split up x,y coordinates into train and test sets
    xy = np.vstack((x, y)).T

    # extract velocity
    v_xy = np.diff(xy, axis=0) / dt

    # kinematics matrices
    X = np.hstack((xy[:-1], v_xy))
    X_train = X[:n_train_samples]
    X_test = X[n_train_samples:]

    # split up neural activity into train and test sets
    Z_train = Y[:n_train_samples]
    Z_test = Y[n_train_samples:-1]

    # fill in the kinematics matrix
    A = np.identity(4)
    b = np.zeros(4)
    # ensure velocity explains positions
    A[0, 2] = dt
    A[1, 3] = dt
    # fit velocity to velocity coefficients
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_train[:-1, 2:4], X_train[1:, 2:4])
    A[2:4, 2:4] = ols.coef_
    # covariance matrix using residuals
    Q = np.zeros(A.shape)
    residuals = X_train[1:] - np.dot(X_train[:-1], A)
    Q_full = np.dot(residuals.T, residuals) / residuals.shape[0]
    Q[2:4, 2:4] = Q_full[2:4, 2:4]

    # kinematics to neural activity matrix
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_train, Z_train)
    C = ols.coef_
    d = ols.intercept_
    # covariance matrix using residuals
    residuals = Z_train - ols.predict(X_train)
    R = np.dot(residuals.T, residuals) / residuals.shape[0]

    # create Kalman Filter
    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=C,
        transition_covariance=Q,
        observation_covariance=R,
        transition_offsets=b,
        observation_offsets=d)

    X_test_hat, _ = kf.filter(Z_test)

    n_outputs = X_test.shape[1]
    r2s = np.zeros(n_outputs)
    corrs = np.zeros(n_outputs)
    # coefficient of determination
    for idx in range(n_outputs):
        r2s[idx] = r2_score(X_test[:, idx], X_test_hat[:, idx])

    # correlation
    for idx in range(n_outputs):
        corrs[idx] = np.corrcoef(X_test[:, idx], X_test_hat[:, idx])[0, 1]

    return X_test, X_test_hat, r2s, corrs