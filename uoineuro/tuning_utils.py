import numpy as np


def calculate_best_frequencies_ecog(tuning_coefs, ecog, return_grid=False):
    if return_grid:
        pref_frequencies = np.zeros((8, 16))
    else:
        pref_frequencies = np.zeros(ecog.n_electrodes)

    for electrode in range(ecog.n_electrodes):
        frequencies, tc = ecog.get_tuning_curve(tuning_coefs=tuning_coefs[electrode])
        pref_frequency = frequencies[np.argmax(tc)]
        if return_grid:
            x, y = ecog.get_xy_for_electrode(electrode)
            pref_frequencies[x, y] = pref_frequency
        else:
            pref_frequencies[electrode] = pref_frequency

    return pref_frequencies
