import numpy as np


def average_encoder(X):
    return sum_encoder(X=X, normalise=True)


def sum_encoder(X, normalise=False):
    if isinstance(X, list):
        X = np.array(X)

    if isinstance(X, np.ndarray):
        if len(X.shape) > 1 and X.shape[0] > 1:
            C = np.sum(X, axis=0) if not normalise else np.average(X, axis=0)
        else:
            C = X.reshape(-1,)
    else:
        C = None

    return C


def max_encoder(X):
    if isinstance(X, list):
        X = np.array(X)

    if isinstance(X, np.ndarray):
        if len(X.shape) > 1 and X.shape[0] > 1:
            C = np.max(X, axis=0)
        else:
            C = X.reshape(-1,)
    else:
        C = None

    return C


def min_encoder(X):
    if isinstance(X, list):
        X = np.array(X)

    if isinstance(X, np.ndarray):
        if len(X.shape) > 1 and X.shape[0] > 1:
            C = np.min(X, axis=0)
        else:
            C = X.reshape(-1,)
    else:
        C = None

    return C


def concatenate_average_max_encoder(X):
    A = average_encoder(X)
    B = max_encoder(X)

    return np.concatenate((A, B))


def concatenate_sum_max_encoder(X):
    A = sum_encoder(X)
    B = max_encoder(X)

    return np.concatenate((A, B))


def concatenate_average_min_encoder(X):
    A = average_encoder(X)
    B = min_encoder(X)

    return np.concatenate((A, B))


def concatenate_sum_min_encoder(X):
    A = sum_encoder(X)
    B = min_encoder(X)

    return np.concatenate((A, B))


def concatenate_min_max_encoder(X):
    A = max_encoder(X)
    B = min_encoder(X)

    return np.concatenate((A, B))