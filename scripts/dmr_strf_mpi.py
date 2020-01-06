"""Calculates STRFs for electrocorticography recordings from rat A1."""
import argparse
import h5py
import numpy as np
import time

from mpi4py import MPI
from pyuoi.linear_model import UoI_Lasso, UoI_ElasticNet
from pyuoi.mpi_utils import Bcast_from_root
from pyuoi.utils import log_likelihood_glm, AIC, BIC
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from uoineuro.tuning_utils import create_strf_design


def main(args):
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    comm = MPI.COMM_WORLD
    rank = comm.rank

    X_train = None
    y_train = None
    n_frames = args.n_frames

    if rank == 0:
        if args.verbose:
            print('Getting data...')

        data = h5py.File(args.data_path, 'r')
        stimulus = data['stim'][:]
        response = data['resp'][:]

        X, Y = create_strf_design(stimulus, response, n_frames)
        y = Y[args.electrode]

        n_samples, n_features = X.shape
        strfs = np.zeros((n_frames, n_features))

        X_train, X_test, y_train, y_test, = \
            train_test_split(X, y,
                             train_size=args.train_frac,
                             random_state=random_state)

        centering = X_train.mean(axis=0, keepdims=True)
        X_train -= centering
        X_test -= centering

    X_train = Bcast_from_root(X_train, comm)
    y_train = Bcast_from_root(y_train, comm)

    if method == 'ridge':
        fitter = RidgeCV(
            normalize=False,
            fit_intercept=False,
            cv=10,
            alphas=np.logspace(3, args.upper_log, num=args.n_alphas)
    elif method == 'elasticnet':
        fitter = ElasticNetCV(l1_ratio=np.array([0.1]),
                              eps=1e-5,
                              n_alphas=100,
                              fit_intercept=False,
                              max_iter=2500,
                              verbose=True,
                              cv=5).fit(X_train, y_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path')
    parser.add_argument('--random_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')
    parser.add_argument('--cell', type=int)
    parser.add_argument('--recording_idx', type=int)
    parser.add_argument('--window_length', type=float, default=0.4)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--stability_selection', type=float, default=1.0)
    parser.add_argument('--estimation_score', default='r2')
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)