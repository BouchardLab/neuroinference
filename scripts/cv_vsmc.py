"""
Performs a cross-validated logistic regression on consonant-vowel syllables
using recordings from the ventral sensory-motor cortex, obtained from the Chang
lab.
"""
import argparse
import h5py
import numpy as np

from pyuoi.linear_model import UoI_L1Logistic
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


def main(args):
    # check random state
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    data_path = args.data_path
    results_path = args.results_path
    results_group = args.results_group
    task = args.task
    n_folds = args.n_folds

    # gather data
    data = h5py.File(data_path, 'r')
    X = data['Xhigh gamma'][:]
    y = data['y'][:]
    data.close()

    if task == 'c':
        X = X[:, :, 75]
        y = np.floor(y / 3)
    elif task == 'v':
        X = X[:, :, 175]
        y = y % 3
    else:
        raise ValueError('Incorrect task.')

    # relevant descriptor variables
    n_samples, n_features = X.shape
    n_outputs = np.unique(y).size
    n_parameters = n_outputs * n_features

    # storage arrays
    logistic_coefs = np.zeros((n_folds, n_outputs, n_features))
    logistic_intercepts = np.zeros((n_folds, n_outputs))
    logistic_scores = np.zeros(n_folds)
    logistic_srs = np.zeros(n_folds)

    uoi_coefs = np.zeros((n_folds, n_outputs, n_features))
    uoi_intercepts = np.zeros((n_folds, n_outputs))
    uoi_scores = np.zeros(n_folds)
    uoi_srs = np.zeros(n_folds)

    train_folds = []
    test_folds = []

    skfold = StratifiedKFold(n_splits=n_folds,
                             shuffle=True,
                             random_state=random_state)

    # iterate over stratified folds
    for fold_idx, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        print('Fold %s' % fold_idx)
        train_folds.append(train_idx)
        test_folds.append(test_idx)

        # preprocess data; importantly, do not standardize
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_center = X_train - X_train.mean(axis=0, keepdims=True)
        X_test_center = X_test - X_train.mean(axis=0, keepdims=True)

        # fit logistic regression
        logistic = LogisticRegressionCV(
            Cs=args.n_Cs,
            cv=args.cv,
            solver='saga',
            penalty='l1',
            multi_class='multinomial',
            max_iter=1000,
            fit_intercept=True)
        logistic.fit(X_train_center, y_train)
        # store results
        logistic_coefs[fold_idx] = logistic.coef_
        logistic_intercepts[fold_idx] = logistic.intercept_
        logistic_srs[fold_idx] = np.count_nonzero(logistic.coef_) / n_parameters
        logistic_scores[fold_idx] = logistic.score(X_test_center, y_test)

        # fit uoi logistic
        uoi = UoI_L1Logistic(
            n_boots_sel=30,
            n_boots_est=30,
            fit_intercept=True,
            standardize=False,
            selection_frac=0.8,
            estimation_frac=0.8,
            n_C=args.n_Cs,
            multi_class='multinomial',
            estimation_score=args.estimation_score,
            estimation_target=args.estimation_target,
            shared_support=args.shared_support,
            random_state=random_state)
        uoi.fit(X_train_center, y_train)
        # store results
        uoi_coefs[fold_idx] = uoi.coef_
        uoi_intercepts[fold_idx] = uoi.intercept_
        uoi_srs[fold_idx] = np.count_nonzero(uoi.coef_) / n_parameters
        uoi_scores[fold_idx] = uoi.score(X_test_center, y_test)

    # collect results
    results = h5py.File(results_path, 'a')
    base = results.require_group(results_group)

    logistic_group = base.require_group('logistic')
    logistic_group['coefs'] = logistic_coefs
    logistic_group['intercepts'] = logistic_intercepts
    logistic_group['scores'] = logistic_scores
    logistic_group['srs'] = logistic_srs

    uoi_group = base.require_group('uoi')
    uoi_group['coefs'] = uoi_coefs
    uoi_group['intercepts'] = uoi_intercepts
    uoi_group['scores'] = uoi_scores
    uoi_group['srs'] = uoi_srs

    folds = base.require_group('folds')
    for idx, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds)):
        folds['train/%s' % idx] = train_fold
        folds['test/%s' % idx] = test_fold

    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--task', default='c')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=-1)

    # fitter arguments
    parser.add_argument('--n_Cs', type=int, default=50)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--shared_support', action='store_true')
    parser.add_argument('--estimation_score', default='acc')
    parser.add_argument('--estimation_target', default='test')

    args = parser.parse_args()

    main(args)
