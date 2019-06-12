from sklearn.utils.extmath import row_norms, safe_sparse_dot
from jitpy.wrapper import jittify

import numpy as np
import math


def equal(x, y):
    equal_value = len(x)
    for i in range(equal_value):
        if x[i].__eq__(y[i]):
            continue
        else:
            return False
    return True


class ngramSet(object):
    def __init__(self, mnemonics, n):
        self.n = n
        self.ngramSet = []
        for i in range(len(mnemonics)):
            mnemonic = mnemonics[i:i+n]
            if len(mnemonic) != n:
                continue
            # delete duplicate n-gram
            # if self.ngramSet.__contains__(mnemonic):
            #     continue
            self.ngramSet.append(mnemonic)


@jittify([list, list, bool], float)
def nh_kernel(X, Y, single=True):
    def nh_similarity(x, y):
        from munkres import Munkres, make_cost_matrix
        import sys

        def _ngram_matrix(set1, set2):
            matrix = []
            for ngram1 in set1:
                unitMatrix = []
                for ngram2 in set2:
                    ngram_similarity = float(longest_common_substring(ngram1, ngram2)) / len(ngram1)
                    if ngram_similarity < 0.7: ngram_similarity = 0
                    unitMatrix.append(100 * ngram_similarity)
                matrix.append(unitMatrix)
            return matrix

        if equal(x, y):
            return 0

        set1 = ngramSet(x, 4).ngramSet
        set2 = ngramSet(y, 4).ngramSet
        _matrix = _ngram_matrix(set1, set2)

        cost_matrix = make_cost_matrix(_matrix, lambda cost: sys.maxint - cost)
        m = Munkres()
        indexes = m.compute(cost_matrix)

        max_matrix = []
        for row, column in indexes:
            value = _matrix[row][column]
            max_matrix.append(value)
        _nh_similarity = np.mean(max_matrix) / 100

        if single:
            return _nh_similarity
        else:
            return 1 - _nh_similarity

    matrix = np.zeros(shape=(len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            matrix[i][j] = nh_similarity(X[i], Y[j])
    return matrix


def longest_common_substring(object1, object2):
    lengths = [[0 for j in range(len(object2) + 1)] for i in range(len(object1) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(object1):
        for j, y in enumerate(object2):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    return lengths[len(object1)][len(object2)]


def ngram_kernel(X, Y, single=True):
    def match(ngram1, ngram2):
        standard = len(ngram1) * 0.7
        _lcs = longest_common_substring(ngram1, ngram2)
        if _lcs < standard:
            return False
        return True

    def ngram_similarity(x, y):
        set1 = ngramSet(x, 4).ngramSet
        set2 = ngramSet(y, 4).ngramSet

        count_of_match = 0
        for ngram1 in set1:
            for ngram2 in set2:
                if match(ngram1, ngram2):
                    count_of_match += 1
                    continue
        _ngram_similarity = float(count_of_match) / len(set1)

        if single:
            return _ngram_similarity
        else:
            return 1 - _ngram_similarity

    matrix = np.zeros(shape=(len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            matrix[i][j] = ngram_similarity(X[i], Y[j])
    return matrix


def lcs_kernel(X, Y, single=True):
    def lcs_similarity(x, y):
        _lcs = longest_common_substring(x, y)
        _lcs_similarity = float(_lcs) / len(x)

        if single:
            return _lcs_similarity
        else:
            return 1 - _lcs_similarity

    matrix = np.zeros(shape=(len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            matrix[i][j] = lcs_similarity(X[i], Y[j])
    return matrix


def ed_kernel(X, Y, single=True):
    def edit_distance(x, y):
        # combined case
        if equal(x, y):
            return 0

        n = len(x)
        m = len(y)
        distance = [[0 for i in range(n + 1)] for i in range(m + 1)]

        if (n == 0): return m
        if (m == 0): return n

        for i in range(n + 1):
            distance[i][0] = i
        for i in range(m + 1):
            distance[0][i] = i

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0
                if x[j - 1].__eq__(y[i - 1]): cost = 1
                distance[i][j] = min(min(distance[i - 1][j] + 1, distance[i][j - 1] + 1), distance[i - 1][j - 1] + cost)
        dissimilarity = float(distance[n][m]) / n

        if single:
            return dissimilarity
        else:
            return 1 - dissimilarity

    matrix = np.zeros(shape=(len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            matrix[i][j] = edit_distance(X[i], Y[j])
    return matrix


def rbf_lcs_kernel(X, Y=None, gamma=None):
    if gamma is None:
        # gamma = 1.0 / X.shape[1]
        gamma = 1.0 / math.pow(2, 5)

    K = lcs_kernel(X, Y, single=False)
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K


def rbf_ngram_kernel(X, Y=None, gamma=None):
    if gamma is None:
        # gamma = 1.0 / X.shape[1]
        gamma = 1.0 / math.pow(2, 5)

    K = ngram_kernel(X, Y, single=False)
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K


def rbf_nh_kernel(X, Y=None, gamma=None):
    if gamma is None:
        # gamma = 1.0 / X.shape[1]
        gamma = 1.0 / math.pow(2, 5)

    K = nh_kernel(X, Y, single=False)
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K


def rbf_ed_kernel(X, Y=None, gamma=None):
    if gamma is None:
        # gamma = 1.0 / X.shape[1]
        gamma = 1.0 / math.pow(2, 5)

    # K = euclidean_distances(X, Y, squared=True)
    K = ed_kernel(X, Y, single=False)
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K


# Kernels
def linear_kernel(X, Y=None):
    """
    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """

    return safe_sparse_dot(X, Y.T, dense_output=True)


def polynomial_kernel(X, Y=None, degree=2, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y::

        K(X, Y) = (gamma <X, Y> + coef0)^degree

    Read more in the :ref:`User Guide <polynomial_kernel>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 3

    gamma : float, default None
        if None, defaults to 1.0 / n_samples_1

    coef0 : int, default 1

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """

    if gamma is None:
        # gamma = 1.0 / X.shape[1]
        gamma = 1.0 / math.pow(2, 5)

    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
    return K
