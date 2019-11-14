"""Calculates STRFs for recordings from mouse retina. Uses the RET1 dataset
from CRCNS."""
import argparse
import h5py
import numpy as np
import time

from mpi4py import MPI
from neuropacks import RET1 as Retina
from pyuoi.linear_model import UoI_Lasso, UoI_Poisson
from pyuoi.mpi_utils import Bcast_from_root
from pyuoi.utils import log_likelihood_glm, AIC, BIC
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score


def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    stimulus_train = None
    response_train = None
    n_frames_per_window = None

    # root will gather the data and initialize storage arrays
    if rank == 0:
        print('Getting data...')
        # gather data if we're in the central rank
        retina = Retina(data_path=args.data_path,
                        random_path=args.random_path)

        stimulus = retina.get_stims_for_recording(
            recording_idx=args.recording_idx,
            window_length=args.window_length)
        response = retina.get_responses_for_recording(
            recording_idx=args.recording_idx,
            window_length=args.window_length,
            cells=args.cell)[:, 0]
        n_frames_per_window = retina.get_n_frames_per_window(
            recording_idx=args.recording_idx,
            window_length=args.window_length)

        # split up data into train and test set
        n_features, n_samples = stimulus.shape
        n_test_samples = int(args.test_frac * n_samples)
        stimulus_test, stimulus_train = np.split(stimulus, [n_test_samples], axis=1)
        response_test, response_train = np.split(response, [n_test_samples])

        # create storage arrays
        strfs = np.zeros((n_frames_per_window, n_features))
        intercepts = np.zeros(n_frames_per_window)
        if 'Lasso' in args.method:
            r2_train = np.zeros(n_frames_per_window)
            r2_test = np.zeros(n_frames_per_window)
        lls = np.zeros(n_frames_per_window)
        aic = np.zeros(n_frames_per_window)
        bic = np.zeros(n_frames_per_window)
        print('Broadcasting data...')

    # broadcast to other ranks
    stimulus_train = Bcast_from_root(stimulus_train, comm)
    response_train = Bcast_from_root(response_train, comm)
    n_frames_per_window = comm.bcast(n_frames_per_window, root=0)

    # iterate over frames in STRF
    for frame in range(n_frames_per_window):
        if args.verbose and rank == 0:
            print('Fitting Frame: ', str(frame))
            t = time.time()

        # obtain fitting procedure
        if args.method == 'Lasso':
            fitter = LassoCV(
                normalize=args.standardize,
                fit_intercept=True,
                cv=5,
                max_iter=10000)

        elif args.method == 'UoI_Lasso':
            fitter = UoI_Lasso(
                standardize=args.standardize,
                n_boots_sel=args.n_boots_sel,
                n_boots_est=args.n_boots_est,
                selection_frac=args.selection_frac,
                estimation_frac=args.estimation_frac,
                n_lambdas=args.n_lambdas,
                stability_selection=args.stability_selection,
                estimation_score=args.estimation_score,
                comm=comm)

        elif args.method == 'UoI_Poisson':
            fitter = UoI_Poisson(
                standardize=args.standardize,
                n_boots_sel=args.n_boots_sel,
                n_boots_est=args.n_boots_est,
                selection_frac=args.selection_frac,
                estimation_frac=args.estimation_frac,
                n_lambdas=args.n_lambdas,
                stability_selection=args.stability_selection,
                estimation_score=args.estimation_score,
                comm=comm)

        else:
            raise ValueError('Method not available.')

        # run the fitting procedure using MPI
        fitter.fit(stimulus_train.T, response_train)

        # root rank will extract fits and calculate scores
        if rank == 0:
            # store the fits
            strfs[frame] = fitter.coef_.T
            intercepts[frame] = fitter.intercept_

            # calculate and store the scores
            y_train_pred = fitter.intercept_ + np.dot(stimulus_train.T, fitter.coef_)
            y_test_pred = fitter.intercept_ + np.dot(stimulus_test.T, fitter.coef_)
            n_selected_features = 1 + np.count_nonzero(fitter.coef_)
            n_samples = y_test_pred.size

            # different scores needed for Lasso/Poisson
            if 'Lasso' in args.method:
                # coefficient of determination
                r2_train[frame] = r2_score(response_train, y_train_pred)
                r2_test[frame] = r2_score(response_test, y_test_pred)
                # training likelihood for ICs
                ll_train = log_likelihood_glm(model='normal',
                                              y_true=response_train,
                                              y_pred=y_train_pred)
                # test likelihood
                lls[frame] = log_likelihood_glm(model='normal',
                                                y_true=response_train,
                                                y_pred=y_test_pred)
            else:
                # training likelihood for ICs
                ll_train = log_likelihood_glm(model='poisson',
                                              y_true=response_train,
                                              y_pred=y_train_pred)
                # test likelihood
                lls[frame] = log_likelihood_glm(model='poisson',
                                                y_true=response_train,
                                                y_pred=y_test_pred)
            # calculate information criteria
            aic[frame] = AIC(ll_train, n_selected_features)
            bic[frame] = BIC(ll_train, n_selected_features, n_samples)

            if args.verbose:
                print('Frame ', frame, 'took ', time.time() - t, ' seconds.')

            # roll back test set window
            response_train = np.roll(response_train, -1)
            response_test = np.roll(response_test, -1)

        # broadcast the rolled response variable
        response_train = Bcast_from_root(response_train, comm)

    # store all results in H5 file
    if rank == 0:
        results = h5py.File(args.results_path, 'a')
        cell_recording = 'cell%s_recording%s' % (args.cell, args.recording_idx)
        group = results.create_group(cell_recording + '/' + args.results_group)
        group['strf'] = strfs
        group['intercepts'] = intercepts
        group['r2_train'] = r2_train
        group['r2_test'] = r2_test
        group['aic'] = aic
        group['bic'] = bic
        results.close()


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
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
