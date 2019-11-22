"""
Performs a cross-validated coupling fit on various neuroscience datasets, using
scikit-learn or Union of Intersections.

This script calculates and stores coupling models on this dataset using a
desired fitting procedure.
"""
import argparse
import h5py
import numpy as np

from neuropacks import ECOG, NHP, PVC11
from pyuoi.linear_model import UoI_Lasso, UoI_Poisson
from pyuoi.utils import log_likelihood_glm, AIC, BIC
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from uoineuro.utils import deviance_poisson


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
    if args.dataset == 'ecog':
        # create data extraction object
        ecog = ECOG(data_path=args.data_path)

        # get response matrix
        Y = ecog.get_response_matrix(
            bounds=(40, 60),
            band=args.band,
            electrodes=None,
            transform=None)
        class_labels = ecog.get_design_matrix(form='id')

    elif args.dataset == 'nhp':
        # create data extraction object
        nhp = NHP(data_path=args.data_path)

        # get response matrix
        Y = nhp.get_response_matrix(
            bin_width=args.bin_width,
            region=args.region,
            transform=args.transform)
        class_labels = np.ones(Y.shape[0])

    elif args.dataset == 'pvc11':
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
    if args.method == 'lasso':
        fitter = LassoCV(
            normalize=args.standardize,
            fit_intercept=True,
            cv=5,
            max_iter=10000)

    elif args.method == 'uoi_lasso':
        fitter = UoI_Lasso(
            standardize=args.standardize,
            n_boots_sel=args.n_boots_sel,
            n_boots_est=args.n_boots_est,
            selection_frac=args.selection_frac,
            estimation_frac=args.estimation_frac,
            n_lambdas=args.n_lambdas,
            stability_selection=args.stability_selection,
            estimation_score=args.estimation_score)

    elif args.method == 'uoi_poisson':
        fitter = UoI_Poisson(
            standardize=args.standardize,
            fit_intercept=True,
            n_lambdas=args.n_lambdas,
            n_boots_sel=args.n_boots_est,
            n_boots_est=args.n_boots_sel,
            selection_frac=args.estimation_frac,
            estimation_frac=args.selection_frac,
            stability_selection=args.stability_selection,
            estimation_score=args.estimation_score,
            solver='lbfgs',
            max_iter=10000,
            warm_start=False)

    # create folds
    skfolds = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=random_state)
    train_folds = {}
    test_folds = {}

    # create storage arrays
    intercepts = np.zeros((n_folds, n_targets))
    coupling_coefs = np.zeros((n_folds, n_targets, n_targets - 1))
    lls_train = np.zeros((n_folds, n_targets))
    lls_test = np.zeros((n_folds, n_targets))
    aics = np.zeros((n_folds, n_targets))
    bics = np.zeros((n_folds, n_targets))
    if 'lasso' in args.method:
        r2s_train = np.zeros((n_folds, n_targets))
        r2s_test = np.zeros((n_folds, n_targets))
    else:
        deviances_train = np.zeros((n_folds, n_targets))
        deviances_test = np.zeros((n_folds, n_targets))

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
            y_true_train = Y_train[:, target]
            y_true_test = Y_test[:, target]

            # perform fit
            if args.method == 'glmnet':
                import glmnet_python
                from cvglmnet import cvglmnet
                from cvglmnetCoef import cvglmnetCoef

                fit = cvglmnet(x=X_train, y=y_true_train, family='poisson',
                               nfolds=n_folds, standardize=standardize)
                coefs = cvglmnetCoef(fit, s='lambda_min').ravel()
                intercept = coefs[0]
                coef = coefs[1:]
            else:
                fitter.fit(X_train, y_true_train)
                intercept = fitter.intercept_
                coef = fitter.coef_
            # store fit
            intercepts[fold_idx, target_idx] = intercept
            coupling_coefs[fold_idx, target_idx] = np.copy(coef)

            # train and test predicted responses
            if 'lasso' in args.method:
                y_pred_train = intercept + np.dot(X_train, coef)
                y_pred_test = intercept + np.dot(X_test, coef)
                model = 'normal'
            else:
                y_pred_train = np.exp(intercept + np.dot(X_train, coef))
                y_pred_test = np.exp(intercept + np.dot(X_test, coef))
                model = 'poisson'

            ll_train = log_likelihood_glm(model=model,
                                          y_true=y_true_train,
                                          y_pred=y_pred_train)
            ll_test = log_likelihood_glm(model=model,
                                         y_true=y_true_test,
                                         y_pred=y_pred_test)
            lls_train[fold_idx, target_idx] = y_pred_train.size * ll_train
            lls_test[fold_idx, target_idx] = y_pred_test.size * ll_test

            n_features = 1 + np.count_nonzero(coef)
            n_train_samples = y_true_train.size
            aics[fold_idx, target_idx] = AIC(ll_train, n_features)
            bics[fold_idx, target_idx] = BIC(ll_train, n_features, n_train_samples)

            # different scores needed for Lasso/Poisson
            if 'lasso' in args.method:
                # coefficient of determination
                r2s_train[fold_idx, target_idx] = r2_score(y_true_train, y_pred_train)
                r2s_test[fold_idx, target_idx] = r2_score(y_true_test, y_pred_test)
            else:
                # calculate information criteria
                deviances_train[fold_idx, target_idx] = deviance_poisson(y_true_train,
                                                                         y_pred_train)
                deviances_test[fold_idx, target_idx] = deviance_poisson(y_true_test,
                                                                        y_pred_test)

    results_file = h5py.File(args.results_path, 'a')
    group = results_file.create_group(args.results_group)
    group['Y'] = Y
    group['intercepts'] = intercepts
    group['coupling_coefs'] = coupling_coefs
    group['lls_train'] = lls_train
    group['lls_test'] = lls_test
    group['aics'] = aics
    group['bics'] = bics
    if 'lasso' in args.method:
        group['r2s_train'] = r2s_train
        group['r2s_test'] = r2s_test
    else:
        group['deviances_train'] = deviances_train
        group['deviances_test'] = deviances_test

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
    parser.add_argument('--method')
    # all datasets
    parser.add_argument('--transform', default=None)
    # nhp arguments
    parser.add_argument('--region', default='M1')
    parser.add_argument('--bin_width', type=float, default=0.25)
    # ecog arguments
    parser.add_argument('--band', default='HG')
    # fitter object arguments
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--random_state', type=int, default=-1)
    # lasso arguments
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=5000)
    # uoi arguments
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--stability_selection', type=float, default=0.95)
    parser.add_argument('--estimation_score', default='r2')
    # other arguments
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
