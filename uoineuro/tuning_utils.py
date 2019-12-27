import numpy as np


def calculate_best_frequencies_ecog(
    tuning_coefs, ecog, return_grid=False, omit_idxs=None
):
    """Calculates the best frequencies for a set of tuning coefficients.

    Parameters
    ----------
    tuning_coefs : array-like, shape (n_coefs,)
        The set of tuning coefficients. Assumes they correspond to a set of
        Gaussian basis functions.

    ecog : neuropack object
        An ECOG neuropack object corresponding to the data.

    return_grid : bool
        If True, the preferred frequencies are returned in the shape of the
        ECoG grid.

    Returns
    -------
    pref_frequencies : float
        The preferred frequency for the set of tuning coefficients.

    """
    # allocate space for frequencies
    if return_grid:
        pref_frequencies = np.zeros((8, 16))
    else:
        pref_frequencies = np.zeros(ecog.n_electrodes)

    for electrode in range(ecog.n_electrodes):
        # extract tuning curve from tuning coefficients
        if electrode in omit_idxs:
            continue

        frequencies = np.linspace(ecog.freq_set[0], ecog.freq_set[-1], 100000)
        _, tc = ecog.get_tuning_curve(tuning_coefs=tuning_coefs[electrode],
                                      frequencies=frequencies)
        # obtain preferred frequency using maximum of tuning curve
        pref_frequency = frequencies[np.argmax(tc)]

        # place preferred frequency in storage depending on desired shape
        if return_grid:
            x, y = ecog.get_xy_for_electrode(electrode)
            pref_frequencies[x, y] = pref_frequency
        else:
            pref_frequencies[electrode] = pref_frequency

    return pref_frequencies


def create_strf_design(stim, resp, n_frames):
    """Create the design and response matrices for STRF data, given a certain
    number of frames.

    Parameters
    ----------
    stim : np.ndarray
        The stimulus matrix.

    resp : np.ndarray
        The response matrix.

    n_frames : int
        The number of frames in the STRF.

    Returns
    -------
    X : np.ndarray
        New design matrix.

    Y : np.ndarray
        New response matrix.
    """
    n_samples, n_features = stim.shape
    n_samples_adj = n_samples - n_frames + 1

    X = np.zeros((n_samples_adj, n_features * n_frames))
    Y = resp[:, (n_frames - 1):]

    for sample in range(n_samples_adj):
        X[sample] = np.flip(stim[sample:sample + n_frames], axis=0).ravel()

    return X, Y
