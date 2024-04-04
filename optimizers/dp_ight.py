import numpy as np


def _epsilon_delta_to_rho(epsilon, delta):
    """Converts (eps, del) to rho for zero-concentrated differential privacy.

    Args: 
        epsilon: privacy parameter
        delta: privacy parameter

    Returns: 
        rho: zCDP privacy parameter
    """
    lg_d = np.log(1 / delta)
    rho = -2 * np.sqrt(epsilon * lg_d + lg_d ** 2) + 2 * lg_d + epsilon
    return rho


def _hard_threshold(w_priv, s):
    """Keeps top-s components of w_priv in terms of absolute value.

    Args:
        w_priv: weight vector
        s: number of components

    Returns:
        w_priv: weight vector with only s nonzero components
    """
    idx_sorted = np.argsort(np.abs(w_priv))[::-1]
    w_priv[idx_sorted[s:]] = 0
    return w_priv


def fit(X, y, grad=None, bound=None, s=None, eta=None, T=None, G=None, epsilon=None, delta=None):
    """Fit function for DP-IGHT.
    
    Args:
        X: dataset
        y: targets, {0, 1} for logistic regression
        grad: gradient function. For linear regression, each term of the loss should be 0.5 * MSE. 
              For logistic regression, each term should be BCE 
        bound: L_2 bound on the norm of the weight vector
        s: maximum number of nonzero components in weight vector
        eta: learning rate
        T: number of iterations
        G: Lipschitz constant of loss. Should be (K*bound + r)K for linear regression and K for logistic 
            regression, where K is the L_2 bound on datapoints and r is the absolute bound on targets
        epsilon: privacy parameter
        delta: privacy parameter

    Returns:
        w_priv: (eps, del)-DP weight vector
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, p = X.shape
    rho = _epsilon_delta_to_rho(epsilon, delta)

    w_priv = np.zeros(p)
    sigma = np.sqrt(T * G ** 2 / (n ** 2 * rho))

    for i in range(T):
        u = np.random.normal(scale=sigma, size=p)
        w_priv = w_priv - eta * (grad(X, y, w_priv) + u)
        w_priv = _hard_threshold(w_priv, s)
        if bound is not None and np.linalg.norm(w_priv, 2) > bound:
            w_priv *= (bound / np.linalg.norm(w_priv, 2))
    
    return w_priv
