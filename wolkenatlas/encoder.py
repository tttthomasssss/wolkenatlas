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