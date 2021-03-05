"""
Performs an experiment testing UoI Logistic on synthetic data.
"""
import argparse
import h5py
import numpy as np
import warnings

from mpi4py import MPI
from pyuoi.linear_model import UoI_L1Logistic
from scipy.stats import truncexpon
from sklearn.linear_model import LogisticRegressionCV


def main(args):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Set up model parameters
    n_features = args.n_features
    n_nz_features = args.n_nz_features
    n_samples = args.n_samples
    # Random number generation
    model_rng = np.random.default_rng(args.model_rng)
    data_rng = np.random.default_rng(args.data_rng)
    # Instantiate parameters and selection profile
    beta = np.zeros(n_features)
    beta[:n_nz_features] = 1
    model_rng.shuffle(beta)
    support = beta != 0
    # Instantiate non-zero parameter values
    nz_beta = truncexpon.rvs(
        b=args.high,
        scale=args.scale,
        size=n_nz_features,
        random_state=model_rng)
    signs = model_rng.choice([-1, 1], size=n_nz_features)
    # Shift the samples and apply the sign mask
    floor = 0.001
    beta[support] = signs * (floor + (args.high - nz_beta))
    intercept = 0
    # Generate data
    X = data_rng.uniform(low=-1, high=1, size=(n_samples, n_features))
    prob = 1. / (1 + np.exp(-(intercept + X @ beta)))
    y = (data_rng.uniform(low=0, high=1, size=n_samples) < prob).astype('int')

    fitter = UoI_L1Logistic(
        n_boots_sel=args.n_boots_sel,
        n_boots_est=args.n_boots_est,
        selection_frac=args.selection_frac,
        estimation_frac=args.estimation_frac,
        n_C=args.n_Cs,
        estimation_score=args.estimation_score,
        stability_selection=args.stability_selection,
        fit_intercept=False,
        standardize=args.standardize,
        max_iter=args.max_iter,
        comm=comm).fit(X, y)

    base = LogisticRegressionCV(
        Cs=args.n_Cs,
        fit_intercept=False,
        cv=5,
        penalty='l1',
        solver='saga',
        tol=1e-4,
        max_iter=args.max_iter).fit(X, y)

    if rank == 0:
        with h5py.File(args.save_path, 'w') as results:
            results['beta'] = beta
            results['intercept'] = np.array([intercept])
            results['beta_uoi'] = fitter.coef_
            results['intercept_uoi'] = np.array([fitter.intercept_])
            results['beta_base'] = base.coef_
            results['intercept_base'] = np.array([base.intercept_])
            results['X'] = X
            results['y'] = y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    # Experiment arguments
    parser.add_argument('--n_features', type=int, default=300)
    parser.add_argument('--n_nz_features', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=1200)
    parser.add_argument('--high', type=int, default=1)
    parser.add_argument('--scale', type=int, default=1)
    # Random number generator arguments
    parser.add_argument('--model_rng', type=int, default=2332)
    parser.add_argument('--data_rng', type=int, default=48119)
    # UoI objects
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--n_Cs', type=int, default=50)
    parser.add_argument('--n_boots_sel', type=int, default=20)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--stability_selection', type=float, default=0.75)
    parser.add_argument('--estimation_score', default='BIC')
    parser.add_argument('--max_iter', type=int, default=10000)

    args = parser.parse_args()

    main(args)
