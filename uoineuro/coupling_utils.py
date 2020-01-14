import itertools
import networkx as nx
import numpy as np

from networkx.algorithms import community
from pyuoi.utils import log_likelihood_glm, AIC, BIC
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import r2_score

from .utils import cosine_similarity, deviance_poisson


def check_metrics(group, fold_idx, unit_idx, metrics, poisson=False):
    """Check that the metrics are correctly calculated.

    Parameters
    ----------
    group : h5py group
        The group in which to check metrics.

    fold_idx : int
        The fold to check.

    unit_idx : int
        The unit to check.

    metrics : list of strings
        The metrics to check.

    poisson : bool
        If True, the fits were generated using a Poisson fitter.
    """

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
        elif metric == 'lls_train' or metric == 'lls_test':
            true = log_likelihood_glm(model=model, y_true=y_true, y_pred=y_pred)
            true *= y_true.size
        elif metric == 'aics':
            ll = log_likelihood_glm(model=model, y_true=y_true, y_pred=y_pred)
            ll *= y_true.size
            n_features = 1 + np.count_nonzero(coupling_coef)
            true = AIC(ll, n_features)
        elif metric == 'bics':
            ll = log_likelihood_glm(model=model, y_true=y_true, y_pred=y_pred)
            ll *= y_true.size
            n_features = 1 + np.count_nonzero(coupling_coef)
            true = BIC(ll, n_features, sample_idx.size)
        elif metric == 'deviances_train' or metric == 'deviances_test':
            true = deviance_poisson(y_true, y_pred)

        assert true == est
    return True


def coupling_coefs_to_weight_matrix(coupling_coefs):
    """Converts a set of coupling coefficients to a weight matrix.

    Parameters
    ----------
    coupling_coefs : np.array
        The set of coupling coefficients.

    Returns
    -------
    weight_matrix : np.array
        A weight matrix, with self connections inputted.
    """
    n_units = coupling_coefs.shape[0]
    weight_matrix = np.zeros((n_units, n_units))

    for unit in range(n_units):
        weight_matrix[unit] = np.insert(coupling_coefs[unit], unit, 0)

    return weight_matrix


def create_directed_graph(coupling_coefs, weighted=False):
    weight_matrix = coupling_coefs_to_weight_matrix(coupling_coefs)

    if not weighted:
        weight_matrix = (weight_matrix != 0).astype('int')

    G = nx.convert_matrix.from_numpy_matrix(
        weight_matrix,
        create_using=nx.DiGraph()
    )
    return G


def create_symmetrized_graph(coupling_coefs, omit_idxs=None, transform=None):
    """Converts a set of coupling coefficients to a symmetrized NetworkX graph.

    Parameters
    ----------
    coupling_coefs : np.array
        The set of coupling coefficients.

    omit_idxs : array-like
        The indices to omit from the graph.

    transform : string or None
        The transformation to apply to the weights.

    Returns
    -------
    G : NetworkX graph
        An undirected graph corresponding to coupling coefs.

    weights_dict : dict
        A dictionary containing all the weights for each pair of units.
    """
    weight_matrix = coupling_coefs_to_weight_matrix(coupling_coefs)
    weights_dict = {}
    n_units = weight_matrix.shape[0]

    G = nx.Graph()

    for unit in range(n_units):
        G.add_node(unit)

    for unit_pair in itertools.combinations(np.arange(n_units), 2):
        u1, u2 = unit_pair

        if (omit_idxs is not None) and ((u1 in omit_idxs) or (u2 in omit_idxs)):
            continue

        weight = 0.5 * (weight_matrix[u1, u2] + weight_matrix[u2, u1])
        if transform == 'square_root':
            weight = np.sign(weight) * np.sqrt(np.abs(weight))

        if weight != 0:
            G.add_weighted_edges_from([(u1, u2, weight)])
            weights_dict[unit_pair] = weight

    return G, weights_dict


def coupling_coef_corrs(coupling_coefs1, coupling_coefs2, correlation='pearson'):
    """Calculate the correlation coefficients between all sets of coupling fits
    for two different procedures.

    Parameters
    ----------
    coupling_coefs1, coupling_coefs2 : np.ndarra
        The coupling coefficients for which to evaluate correlations.

    correlation : string
        The type of correlation to calculate.

    Returns
    -------
    correlations : np.ndarray
        The correlations between each set of coupling fits.
    """
    n_neurons = coupling_coefs1.shape[0]
    correlations = np.zeros(n_neurons)

    for neuron in range(n_neurons):
        ccs1 = coupling_coefs1[neuron]
        ccs2 = coupling_coefs2[neuron]

        if np.array_equal(ccs1, ccs2):
            correlations[neuron] = 1.
        elif np.all(ccs1 == 0) or np.all(ccs2 == 0):
            correlations[neuron] = 0
        else:
            if correlation == 'pearson':
                correlations[neuron] = np.corrcoef(ccs1, ccs2)[0, 1]
            elif correlation == 'spearman':
                correlations[neuron] = spearmanr(ccs1, ccs2).correlation
            elif correlation == 'cosine':
                correlations[neuron] = cosine_similarity(ccs1, ccs2)

    return correlations


def selection_profiles_by_chance(true, compare):
    """Calculate the probability that the selection profile of dataset2 would
    match up with the selection profile of dataset1 according to the
    hypergeometric distribution.

    Parameters
    ----------
    true, compare : np.ndarra
        The coupling coefficients for which to evaluate the hypergeometric
        test.

    Returns
    -------
    probabilities : np.ndarray
        The probability of matching selection profiles under the hypergeometric
        distribution.
    """
    n_neurons, M = true.shape
    probabilities = np.zeros(n_neurons)

    for neuron in range(n_neurons):
        n = np.count_nonzero(true[neuron])
        N = np.count_nonzero(compare[neuron])
        rv = hypergeom(M=M, n=n, N=N)

        overlap = np.count_nonzero(true[neuron] * compare[neuron])
        probabilities[neuron] = 1 - rv.cdf(x=overlap)

    return probabilities


def compute_modularity(G):
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected(reciprocal=True)
    
    community_detection = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, community_detection)
    return modularity


def compute_controllability_curve(G, times=None, metric='mineig'):
    traces = np.zeros(Ts.shape)
    for idx, T in enumerate(Ts):
        C = np.zeros((A.shape[0], A.shape[0] * T))
        for ii in range(T):
            C[:, A.shape[0]*ii:A.shape[0]*(ii + 1)] = np.linalg.matrix_power(A, ii)
        W = np.dot(C, C.T)
        traces[idx] = np.min(np.linalg.eigh(W)[0])
    return G