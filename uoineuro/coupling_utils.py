import itertools
import networkx as nx
import numpy as np

from pyuoi.utils import log_likelihood_glm, AIC, BIC
from sklearn.metrics import r2_score


def check_metrics(group, fold_idx, unit_idx, metrics, poisson=False):
    """Check that the metrics are correctly calculated."""

    # gather datasets
    Y = group['Y']
    X = np.delete(Y, unit_idx, axis=1)
    y = Y[:, unit_idx]
    intercept = group['intercepts'][fold_idx, unit_idx]
    coupling_coef = group['coupling_coefs'][fold_idx, unit_idx, :]

    for metric in metrics:
        if ('train' in metric) or ('aic' in metric) or ('bic' in metric):
            sample_idx = group['train_folds/fold_' + str(fold_idx)][:]
        else:
            sample_idx = group['test_folds/fold_' + str(fold_idx)]

        y_true = y[sample_idx]
        X_sub = X[sample_idx]
        if poisson:
            model = 'poisson'
            y_pred = np.exp(intercept + np.dot(X_sub, coupling_coef))
        else:
            model = 'normal'
            y_pred = intercept + np.dot(X_sub, coupling_coef)

        est = group[metric][fold_idx, unit_idx]

        if metric == 'r2s_train' or metric == 'r2s_test':
            true = r2_score(y_true, y_pred)
            assert true == est
        elif metric == 'lls_train' or metric == 'lls_test':
            true = log_likelihood_glm(model=model, y_true=y_true, y_pred=y_pred)
        elif metric == 'aics':
            ll = log_likelihood_glm(model=model, y_true=y_true, y_pred=y_pred)
            n_features = 1 + np.count_nonzero(coupling_coef)
            true = AIC(ll, n_features)
        elif metric == 'bics':
            ll = log_likelihood_glm(model=model, y_true=y_true, y_pred=y_pred)
            n_features = 1 + np.count_nonzero(coupling_coef)
            true = BIC(ll, n_features, sample_idx.size)

        assert true == est
    return True


def coupling_coefs_to_weight_matrix(coupling_coefs):
    n_units = coupling_coefs.shape[0]
    weight_matrix = np.zeros((n_units, n_units))

    for unit in range(n_units):
        weight_matrix[unit] = np.insert(coupling_coefs[unit], unit, 0)

    return weight_matrix


def create_symmetrized_graph(coupling_coefs, omit_idxs=None, transform=None):
    weight_matrix = coupling_coefs_to_weight_matrix(coupling_coefs)
    weights_dict = {}
    n_units = weight_matrix.shape[0]

    G = nx.Graph()

    for unit in range(n_units):
        G.add_node(unit)

    for unit_pair in itertools.combinations(np.arange(n_units), 2):
        u1, u2 = unit_pair

        if (u1 in omit_idxs) or (u2 in omit_idxs):
            continue

        weight = 0.5 * (weight_matrix[u1, u2] + weight_matrix[u2, u1])
        if transform == 'square_root':
            weight = np.sign(weight) * np.sqrt(np.abs(weight))

        if weight != 0:
            G.add_weighted_edges_from([(u1, u2, weight)])
            weights_dict[unit_pair] = weight

    return G, weights_dict
