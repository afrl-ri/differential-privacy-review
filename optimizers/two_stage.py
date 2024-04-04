from .dp_ight import _hard_threshold
from .utils import mse_support
import numpy as np


def _generate_support(X, y, supp, s, k, big_lambd, epsilon):
    """Identify features to include in final solution. 
    
    Args:
        X: dataset
        y: targets
        supp: function to determine important features
        s: number of features
        k: number of splits
        big_lambd: multiplicand for L_1 regularization
        epsilon: privacy parameter

    Return:
        g: vector of 0s and 1s corresponding to locations selected for final solution
    """
    g = np.zeros(X.shape[1])
    for i in range(k):
        start = int(i / k * X.shape[0])
        end = int((i + 1) / k * X.shape[0])
        X_i = X[start:end, :]
        y_i = y[start:end]
        v_i = supp(X_i, y_i, s, big_lambd)
        g += 1/k * v_i
    g += np.random.laplace(scale=2 * s / (k * epsilon), size=X.shape[1])
    g = _hard_threshold(g, s)
    g = np.where(g != 0, 1, 0)
    return g


def fit(X, y, supp=None, perturbed_optim=None, s=None, big_lambd=None, small_lambd=None, 
        big_delta=None, xi=None, epsilon=None, delta=None):
    """Fit function for two-stage. 
    
    Args:
        X: dataset. Scale so that its max absolute value is 1
        y: targets. Scale so that its max absolute value is s
        supp: function for support selection
        perturbed_optim: function for perturbed optimization
        s: number of nonzero coefficients
        big_lambd: multiplicand for L_1 regularization
        small_lambd: maximum eigenvalue. For linear regression, s. For logistic regresison, 0.25s
        big_delta: multiplicand for L_2 regularization
        xi: Lipschitz constant. 2s^(3/2) for linear regression, s^(1/2) for logistic regression
        epsilon: privacy parameter
        delta: privacy parameter

    Returns: 
        out: (eps, del)-DP weight vector
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    if supp == mse_support: 
        small_lambd = s
        big_delta = 2 * small_lambd / epsilon
        xi = 2 * s ** (3 / 2)
    else:
        small_lambd = 0.25 * s
        big_delta = 2 * small_lambd / epsilon
        xi = s ** (1/2)
    n, p = X.shape

    support = _generate_support(X, y, supp, s, int(np.sqrt(n)), big_lambd, epsilon / 2)
    
    y_new = np.where(y < -s, -s, y)
    y_new = np.where(y_new > s, s, y_new)
    X_top_s = np.partition(np.abs(X), -s)
    X_top_s = X_top_s[:, -s:]
    norms = np.linalg.norm(X_top_s, axis=1)[:, None]
    X = np.where(norms < np.sqrt(s), X, np.sqrt(s) * X / norms)

    out = perturbed_optim(X, y, s, xi, small_lambd, big_delta, support, epsilon, delta)
    return out


    