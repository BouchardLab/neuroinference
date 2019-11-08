"""
Performs a cross-validated coupling fit on various neuroscience datasets, using
sci-kit learn or UoI.

This script calculates and stores coupling models on this dataset using a
desired fitting procedure.
"""
import argparse
import h5py
import numpy as np

from neuropacks import ECOG, NHP, PVC11
from sem import SEMSolver


def main(args):
    # check random state
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    if args.dataset == 'ECOG':
        # create data extraction object
        ecog = ECOG(data_path=args.data_path)

        # get response matrix
        Y = ecog.get_response_matrix(
            bounds=(40, 60),
            band=args.band,
            electrodes=None,
            transform=None
        )
        class_labels = ecog.get_design_matrix(form='id')

    elif args.dataset == 'NHP':
        # create data extraction object
        nhp = NHP(data_path=args.data_path)

        # get response matrix
        Y = nhp.get_response_matrix(
            bin_width=args.bin_width,
            region=args.region,
            transform=args.transform
        )
        class_labels = None

    elif args.dataset == 'PVC11':
        # create data extraction object
        pvc = PVC11(data_path=args.data_path)

        # get response matrix
        Y = pvc.get_response_matrix(
            transform=args.transform
        )
        class_labels = pvc.get_design_matrix(form='label')

    else:
        raise ValueError('Dataset not available.')

    # clear out empty units
    Y = Y[:, np.argwhere(Y.sum(axis=0) != 0).ravel()]

    # create solver object
    solver = SEMSolver(Y=Y)

    # create args
    sem_args = {
        'model': 'c',
        'method': args.method,
        'class_labels': class_labels,
        'targets': None,
        'n_folds': args.n_folds,
        'random_state': random_state,
        'verbose': args.verbose,
        'fit_intercept': True,
        'max_iter': args.max_iter,
        'metrics': ['r2', 'AIC', 'BIC'],
    }

    if args.method == 'Lasso':
        sem_args['normalize'] = args.normalize
        sem_args['cv'] = args.cv

    elif 'UoI' in args.method:
        # important: we use normalize/standardize to mean the same thing
        sem_args['standardize'] = args.normalize
        sem_args['n_boots_sel'] = args.n_boots_sel
        sem_args['n_boots_est'] = args.n_boots_est
        sem_args['selection_frac'] = args.selection_frac
        sem_args['estimation_frac'] = args.estimation_frac
        sem_args['n_lambdas'] = args.n_lambdas
        sem_args['stability_selection'] = args.stability_selection
        sem_args['estimation_score'] = args.estimation_score

    else:
        raise ValueError('Method is not valid.')

    # Poisson specific parameters
    if args.method == 'UoI_Poisson':
        sem_args['solver'] = args.solver
        sem_args['metrics'] = ['AIC', 'BIC']

    # perform cross-validated coupling fits
    results = solver.estimation(**sem_args)

    results_file = h5py.File(args.results_path, 'a')
    group = results_file.create_group(args.results_group)
    group['Y'] = Y
    # place results in group
    for key in results.keys():
        # need to handle training and test folds separately
        if key == 'training_folds' or key == 'test_folds':
            folds_group = group.create_group(key)
            for fold_key, fold_val in results[key].items():
                folds_group[fold_key] = fold_val
        else:
            group[key] = results[key]

    results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--dataset')
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')

    # All datasets
    parser.add_argument('--transform', default=None)

    # NHP arguments
    parser.add_argument('--region', default='M1')
    parser.add_argument('--bin_width', type=float, default=0.25)

    # ECOG arguments
    parser.add_argument('--band', default='HG')

    # fitter object arguments
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--random_state', type=int, default=-1)

    # LassoCV
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=5000)

    # UoI arguments
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--stability_selection', type=float, default=1.)
    parser.add_argument('--estimation_score', default='r2')

    # UoI Poisson arguments
    parser.add_argument('--solver', default='lbfgs')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
