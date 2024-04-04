import numpy as np
from scipy import sparse
from optimizers.ht_frank_wolfe import exponential

def fit(X, y, grad=None, lambd=None, K=None, T=None, epsilon=None, delta=None):
    """Heavy-Tailed Private Lasso
    
    Args:
        X: Dataset. Can be L_1 norm bounded to 1. 
        y: targets. Can be L_1 norm bounded to 1. 
        grad: gradient function. 
        lambd: maximum L_1 norm of the weight vector. 
        K: Cutoff for data and targets. 
        T: number of iterations. 
        eps: privacy parameter. 
        delta: privacy parameter. 
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, d = X.shape
    w = np.zeros(d)
    X = np.sign(X) * np.minimum(np.abs(X), K)
    y = np.sign(y) * np.minimum(np.abs(y), K)

    constraint = lambd * sparse.vstack(
        (sparse.identity(d, format="csc"), -sparse.identity(d, format="csc")),
        format="csc",
    )

    for t in range(1, T + 1):
        g = 2 * grad(X, y, w)
        scores = -constraint @ g
        exp = exponential(scores, 16 * lambd * K ** 2 / n, epsilon / (2 * np.sqrt(2 * T * np.log(1 / delta))))
        eta = 2 / (t + 2)
        w = (1 - eta) * w + eta * constraint[exp, :]
        w = np.squeeze(np.asarray(w))
    return w