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
    coefs = np.zeros((n_folds, n_outputs, n_features))
    intercepts = np.zeros((n_folds, n_outputs))
    scores = np.zeros(n_folds)
    selection_ratios = np.zeros(n_folds)

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
        if args.method == 'logistic':
            fitter = LogisticRegressionCV(
                Cs=args.n_Cs,
                cv=args.cv,
                solver='saga',
                penalty='l1',
                multi_class='multinomial',
                max_iter=1000,
                fit_intercept=True)
        elif args.method == 'uoi':
            fitter = UoI_L1Logistic(
                n_boots_sel=args.n_boots_sel,
                n_boots_est=args.n_boots_est,
                fit_intercept=True,
                standardize=False,
                selection_frac=args.selection_frac,
                estimation_frac=args.estimation_frac,
                n_C=args.n_Cs,
                multi_class='multinomial',
                estimation_score=args.estimation_score,
                shared_support=args.shared_support,
                random_state=random_state)

        # store results
        fitter.fit(X_train_center, y_train)
        # store results
        coefs[fold_idx] = fitter.coef_
        intercepts[fold_idx] = fitter.intercept_
        selection_ratios[fold_idx] = np.count_nonzero(fitter.coef_) / n_parameters
        scores[fold_idx] = fitter.score(X_test_center, y_test)

    # collect results
    results = h5py.File(results_path, 'a')
    group = results.require_group(results_group)
    group['coefs'] = coefs
    group['intercepts'] = intercepts
    group['scores'] = scores
    group['selection_ratios'] = selection_ratios
    # store folds
    folds = group.require_group('folds')
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
    parser.add_argument('--method')
    parser.add_argument('--task', default='c')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=-1)
    # logistic arguments
    parser.add_argument('--n_Cs', type=int, default=50)
    parser.add_argument('--cv', type=int, default=5)
    # uoi arguments
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--stability_selection', type=float, default=0.95)
    parser.add_argument('--estimation_score', default='acc')

    args = parser.parse_args()

    main(args)
