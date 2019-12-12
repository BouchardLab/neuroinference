import argparse
import h5py
import numpy as np

from neuropacks import NHP
from uoineuro.dimensionality_reduction_utils import (apply_linear_decoder,
                                                     apply_kalman_filter)


def main(args):
    data_path = args.data_path
    results_path = args.results_path
    dt = args.bin_width
    train_frac = args.train_frac

    left_idx = args.left_idx
    if args.right_idx == 0:
        right_idx = None
    else:
        right_idx = args.right_idx

    # extract neural responses
    results = h5py.File(results_path, 'r')
    Y = results['data/Y'][:]
    results.close()

    # get position of hand
    nhp = NHP(data_path=data_path)
    positions = nhp.get_binned_positions(bin_width=dt)
    positions = positions[left_idx:right_idx, :]
    x = positions[:, 0]
    y = positions[:, 1]

    # create k array
    max_ks = np.arange(args.min_max_k, args.max_max_k, args.max_k_spacing)
    n_max_ks = max_ks.size
    reps = args.reps

    # decode with kalman filter using all columns
    _, _, base_r2s, base_corrs = apply_kalman_filter(
        x, y, Y, dt=dt, train_frac=train_frac
    )

    uoi_kalman_corrs = np.zeros((reps, n_max_ks, 4))
    uoi_kalman_r2s = np.zeros((reps, n_max_ks, 4))
    uoi_linear_corrs = np.zeros((reps, n_max_ks, 2))
    uoi_linear_r2s = np.zeros((reps, n_max_ks, 2))
    css_kalman_corrs = np.zeros((reps, n_max_ks, 4))
    css_kalman_r2s = np.zeros((reps, n_max_ks, 4))
    css_linear_corrs = np.zeros((reps, n_max_ks, 2))
    css_linear_r2s = np.zeros((reps, n_max_ks, 2))

    # iterate over repetitions
    for rep in range(reps):
        if args.verbose:
            print('Repetition ', str(rep))
        # iterate over ranks
        for k_idx, max_k in enumerate(max_ks):
            if args.verbose:
                print('    k: ', str(max_k))
            # read out the columns
            results = h5py.File(results_path, 'r')
            uoi_c = results['uoi/columns/%s/%s' % (rep, max_k)][:]
            css_c = results['css/columns/%s/%s' % (rep, max_k)][:]
            results.close()

            # apply kalman filter to uoi columns
            _, _, r2s, corrs = apply_kalman_filter(
                x, y, Y[:, uoi_c], dt=dt, train_frac=train_frac
            )
            uoi_kalman_corrs[rep, k_idx] = corrs
            uoi_kalman_r2s[rep, k_idx] = r2s

            # apply linear decoder to uoi columns
            _, _, r2s, corrs = apply_linear_decoder(
                x, y, Y[:, uoi_c], train_frac=train_frac
            )
            uoi_linear_corrs[rep, k_idx] = corrs
            uoi_linear_r2s[rep, k_idx] = r2s

            # apply kalman filter to to css columns
            _, _, r2s, corrs = apply_kalman_filter(
                x, y, Y[:, css_c], dt=dt, train_frac=train_frac
            )
            css_kalman_corrs[rep, k_idx] = corrs
            css_kalman_r2s[rep, k_idx] = r2s

            # apply linear decoder to css columns
            _, _, r2s, corrs = apply_linear_decoder(
                x, y, Y[:, css_c], train_frac=train_frac
            )
            css_linear_corrs[rep, k_idx] = corrs
            css_linear_r2s[rep, k_idx] = r2s

    results = h5py.File(results_path, 'a')
    results['base_scores'] = base_corrs
    results['base_r2s'] = base_r2s
    results['uoi/scores/kalman_corrs'] = uoi_kalman_corrs
    results['uoi/scores/kalman_r2s'] = uoi_kalman_r2s
    results['uoi/scores/linear_corrs'] = uoi_linear_corrs
    results['uoi/scores/linear_r2s'] = uoi_linear_r2s
    results['css/scores/kalman_corrs'] = css_kalman_corrs
    results['css/scores/kalman_r2s'] = css_kalman_r2s
    results['css/scores/linear_corrs'] = css_linear_corrs
    results['css/scores/linear_r2s'] = css_linear_r2s
    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--bin_width', type=float, default=0.25)
    parser.add_argument('--left_idx', type=int, default=0)
    parser.add_argument('--right_idx', type=int, default=0)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--min_max_k', type=int, default=2)
    parser.add_argument('--max_max_k', type=int, default=100)
    parser.add_argument('--max_k_spacing', type=int, default=2)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
