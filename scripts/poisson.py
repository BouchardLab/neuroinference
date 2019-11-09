"""
Performs a cross-validated coupling fit on various neuroscience datasets,
using a Poisson fitter. Can use either glmnet or UoI.

This script performs and stores coupling models on this dataset using a
desired fitting procedure.
"""
import argparse
import h5py
import glmnet_python
import numpy as np

from utils import log_likelihood, deviance, AIC, BIC

from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from neuropacks import ECOG, NHP, PVC11
from pyuoi.linear_model import UoI_Poisson
from sklearn.model_selection import StratifiedKFold


def main(args):
    # check random state
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    n_folds = args.n_folds
    standardize = args.standardize
    verbose = args.verbose

    # extract dataset
    if args.dataset == 'ECOG':
        # create data extraction object
        ecog = ECOG(data_path=args.data_path)

        # get response matrix
        Y = ecog.get_response_matrix(
            bounds=(40, 60),
            band=args.band,
            electrodes=None,
            transform=None)
        class_labels = ecog.get_design_matrix(form='id')

    elif args.dataset == 'NHP':
        # create data extraction object
        nhp = NHP(data_path=args.data_path)

        # get response matrix
        Y = nhp.get_response_matrix(
            bin_width=args.bin_width,
            region=args.region,
            transform=args.transform)
        class_labels = np.ones(Y.shape[0])

    elif args.dataset == 'PVC11':
        # create data extraction object
        pvc = PVC11(data_path=args.data_path)

        # get response matrix
        Y = pvc.get_response_matrix(transform=args.transform)
        class_labels = pvc.get_design_matrix(form='label')

    else:
        raise ValueError('Dataset not available.')

    # clear out empty units
    Y = Y[:, np.argwhere(Y.sum(axis=0) != 0).ravel()]
    n_targets = Y.shape[1]
    targets = np.arange(n_targets)

    # create fitter
    if args.fitter == 'UoI_Poisson':
        fitter = UoI_Poisson(
            n_lambdas=args.n_lambdas,
            n_boots_sel=args.n_boots_est,
            n_boots_est=args.n_boots_sel,
            selection_frac=args.estimation_frac,
            estimation_frac=args.selection_frac,
            stability_selection=args.stability_selection,
            estimation_score=args.estimation_score,
            solver='lbfgs',
            standardize=standardize,
            fit_intercept=True,
            max_iter=10000,
            warm_start=False)

    # create folds
    skfolds = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=random_state)
    train_folds = {}
    test_folds = {}

    # create results dict
    intercepts = np.zeros((n_folds, n_targets))
    coupling_coefs = np.zeros((n_folds, n_targets, n_targets - 1))
    train_lls = np.zeros((n_folds, n_targets))
    test_lls = np.zeros((n_folds, n_targets))
    deviances_train = np.zeros((n_folds, n_targets))
    deviances_test = np.zeros((n_folds, n_targets))
    AICs = np.zeros((n_folds, n_targets))
    BICs = np.zeros((n_folds, n_targets))

    # outer loop: create and iterate over cross-validation folds
    for fold_idx, (train_idx, test_idx) in enumerate(
        skfolds.split(y=class_labels, X=class_labels)
    ):
        if verbose:
            print('Fold %s' % fold_idx, flush=True)

        train_folds['fold_%s' % fold_idx] = train_idx
        test_folds['fold_%s' % fold_idx] = test_idx

        Y_train = Y[train_idx, :]
        Y_test = Y[test_idx, :]

        # inner loop: iterate over neuron fits
        for target_idx, target in enumerate(targets):
            if verbose:
                print('Target ', target)

            # training design and response matrices
            X_train = np.delete(Y_train, target, axis=1)
            X_test = np.delete(Y_test, target, axis=1)
            y_train = Y_train[:, target]
            y_test = Y_test[:, target]

            # perform fit
            if args.fitter == 'glmnet':
                fit = cvglmnet(x=X_train, y=y_train, family='poisson',
                               nfolds=n_folds, standardize=standardize)
                coefs = cvglmnetCoef(fit, s='lambda_min').ravel()
                intercept = coefs[0]
                coef = coefs[1:]
            else:
                fitter.fit(X_train, y_train)
                intercept = fitter.intercept_
                coef = fitter.coef_

            intercepts[fold_idx, target_idx] = intercept
            coupling_coefs[fold_idx, target_idx] = np.copy(coef)

            # test design and response matrices
            y_pred_train = np.exp(intercept + np.dot(X_train, coef))
            y_pred_test = np.exp(intercept + np.dot(X_test, coef))

            # metrics
            train_lls[fold_idx, target_idx] = log_likelihood(y_train, y_pred_train)
            test_lls[fold_idx, target_idx] = log_likelihood(y_test, y_pred_test)
            deviances_train[fold_idx, target_idx] = deviance(y_train, y_pred_train)
            deviances_test[fold_idx, target_idx] = deviance(y_test, y_pred_test)
            n_features = 1 + np.count_nonzero(coef)
            AICs[fold_idx, target_idx] = AIC(y_train, y_pred_train, n_features)
            BICs[fold_idx, target_idx] = BIC(y_train, y_pred_train, n_features)

    results_file = h5py.File(args.results_path, 'a')
    group = results_file.create_group(args.results_group)
    group['Y'] = Y
    group['intercepts'] = intercepts
    group['coupling_coefs'] = coupling_coefs
    group['train_lls'] = train_lls
    group['test_lls'] = test_lls
    group['deviances_train'] = deviances_train
    group['deviances_test'] = deviances_test
    group['AICs'] = AICs
    group['BICs'] = BICs

    train_folds_group = group.create_group('train_folds')
    test_folds_group = group.create_group('test_folds')

    for fold_key, fold_val in train_folds.items():
        train_folds_group[fold_key] = fold_val
    for fold_key, fold_val in test_folds.items():
        test_folds_group[fold_key] = fold_val

    results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--dataset')
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--fitter')

    # All datasets
    parser.add_argument('--transform', default=None)

    # NHP arguments
    parser.add_argument('--region', default='M1')
    parser.add_argument('--bin_width', type=float, default=0.25)

    # ECOG arguments
    parser.add_argument('--band', default='HG')

    # fitter object arguments
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--random_state', type=int, default=-1)

    # UoI arguments
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--stability_selection', type=float, default=0.95)
    parser.add_argument('--estimation_score', default='log')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
