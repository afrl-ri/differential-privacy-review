import numpy as np
import scipy as sp
from optimizers.ht_sparse_lasso import peeling

def correction_constant(m, sig, beta):
    F = lambda x: sp.stats.norm.cdf(x, loc=0, scale=1 / np.sqrt(beta))
    first = 2 * np.sqrt(2) / 3 * (F((-np.sqrt(2) + m) / sig) - \
                                  F((-np.sqrt(2) - m) / sig))
    second = -(m - m**3 / 6) * (F((-np.sqrt(2) + m) / sig) - \
                                  F((-np.sqrt(2) - m) / sig))
    third = sig * (1 - m ** 2 / 2) / np.sqrt(2 * np.pi) * \
                (np.exp(-0.5 * (((np.sqrt(2) + m) / sig) ** 2)) - \
                 np.exp(-0.5 * (((np.sqrt(2) - m) / sig) ** 2)))
    fourth = m * sig ** 2 / 2 * (F((-np.sqrt(2) - m) / sig) + \
                                 F((-np.sqrt(2) + m) / sig) + \
                                 1 / np.sqrt(2 * np.pi) * (
        (np.sqrt(2) + m) / sig * np.exp(-0.5 * (((np.sqrt(2) + m) / sig) ** 2)) + \
        (np.sqrt(2) - m) / sig * np.exp(-0.5 * (((np.sqrt(2) - m) / sig) ** 2))))
    fifth = sig ** 3 / ( 6 * np.sqrt(2 * np.pi)) * \
        (((((np.sqrt(2) - m) / sig) ** 2) + 2) * np.exp(-0.5 * (((np.sqrt(2) - m) / sig) ** 2)) - \
         ((((np.sqrt(2) + m) / sig) ** 2) + 2) * np.exp(-0.5 * (((np.sqrt(2) + m) / sig) ** 2)))
    return first + second + third + fourth + fifth


def robust_grad(X, y, w, grad, k, beta):
    peg = grad(X, y, w, per_example=True)
    peg = peg.astype(float)
    first = peg * (1 - peg ** 2 / (2 * k ** 2 * beta))
    second = peg ** 3 / (6 * k ** 2)
    third = correction_constant(peg / k, np.abs(peg) / (k * np.sqrt(beta)) + 1e-6, beta)
    return np.mean(first - second, axis=0) + k * np.mean(third, axis=0)


def fit(X, y, grad=None, s=None, beta=None, k=None, T=None, eta=None, epsilon=None, delta=None):
    """Heavy-Tailed Sparse Optimizer
    
    Args:
        X: dataset. 
        y: targets. 
        grad: gradient function. 
        s: sparsity parameter. 
        beta: Scaling factor for robust gradient. Can be set to 1. 
        k: scaling factor for robust gradient. 
        T: iteration parameter. 
        eta: learning rate. 
        eps: privacy parameter. 
        delta: privacy parameter. 
        
    Returns: 
        w: weight vector
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, d = X.shape
    splits = np.linspace(0, n, T + 1).astype(int)
    w = np.zeros(d)
    for t in range(1, T + 1):
        start = splits[t - 1]
        end = splits[t]
        rg = robust_grad(X[start:end, :], y[start:end], w, grad, k, beta)
        w = w - eta * rg
        w = peeling(w, s, epsilon, delta, 4 * k * np.sqrt(2) * eta / (end - start))
    
    return w