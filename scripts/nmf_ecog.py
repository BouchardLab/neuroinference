import argparse
import h5py
import numpy as np

from neuropacks import ECOG
from pyuoi.decomposition import UoI_NMF
from sklearn.decomposition import NMF
from uoineuro.dimensionality_reduction_utils import bi_cross_validator


def main(args):
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    ecog = ECOG(data_path=args.data_path)
    Y = ecog.get_response_matrix(
        bounds=(40, 60),
        band='HG',
        electrodes=None,
        transform=None)
    stratify = ecog.get_design_matrix(form='id')

    dead_indices = [0, 19, 21]
    Y[:, dead_indices] = 0
    Y = Y + np.abs(np.min(Y, axis=0, keepdims=True))

    if args.method == 'NMF':
        nmf = NMF(init='random',
                  solver=args.solver,
                  beta_loss='frobenius',
                  max_iter=1000,
                  alpha=args.alpha,
                  l1_ratio=args.l1_ratio)
        ranks = np.arange(1, args.max_rank + 1)
        # run bi-cross-validator on data
        k_hat, errors = bi_cross_validator(Y, nmf, ranks,
                                           row_frac=args.row_frac,
                                           col_frac=args.col_frac,
                                           n_reps=args.n_reps,
                                           stratify=stratify,
                                           random_state=random_state,
                                           verbose=args.verbose)
        results = h5py.File(args.results_path, 'r')
        group = results.create_group(args.results_group)
        group['errors'] = errors
        results.close()
    else:
        uoi = UoI_NMF(n_boots=args.n_boots,
                      ranks=np.arange(2, args.max_rank + 1),
                      nmf_init='random',
                      nmf_solver=args.solver,
                      nmf_beta_loss='frobenius',
                      nmf_max_iter=1000)
        uoi.fit(Y)
        results = h5py.File(args.results_path, 'r')
        group = results.create_group(args.results_group)
        group['components_'] = uoi.components_
        group['dissimilarity'] = uoi.dissimilarity_
        group['bases_samples'] = uoi.bases_samples_
        group['bases_labels'] = uoi.bases_samples_labels_
        results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')
    # bi-cross-validation arguments
    parser.add_argument('--row_frac', type=float, default=0.75)
    parser.add_argument('--col_frac', type=float, default=0.75)
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=-1)
    parser.add_argument('--verbose', store_action=True)
    # nmf arguments
    parser.add_argument('--solver', default='cd')
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--l1_ratio', type=float, default=0.)

    args = parser.parse_args()
    main(args)
