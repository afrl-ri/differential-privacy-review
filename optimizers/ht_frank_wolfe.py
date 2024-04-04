import numpy as np
from scipy import sparse
from optimizers.ht_sparse_opt import robust_grad

def exponential(score, sensitivity, eps):
    probs = np.exp(eps * score / (2 * sensitivity))
    probs = probs / np.linalg.norm(probs, ord=1)
    return np.random.choice(score.shape[0], p=probs)

def fit(X, y, grad=None, lambd=None, s=None, T=None, beta=None, epsilon=None, delta=None):
    """Heavy-Tailed Sparse Optimizer
    
    Args:
        X: dataset. 
        y: targets. 
        grad: gradient function. 
        lambd: L_1 norm constraint. 
        s: scaling parameter. 
        T: number of iterations. 
        beta: Scaling factor for robust gradient. Can be set to 1. 
        eps: privacy parameter. 
        
    Returns: 
        w: weight vector
    """
    n, d = X.shape
    w = np.zeros(d)
    splits = np.linspace(0, n, T + 1).astype(int)

    constraint = lambd * sparse.vstack(
        (sparse.identity(d, format="csc"), -sparse.identity(d, format="csc")),
        format="csc",
    )

    for t in range(1, T + 1):
        start = splits[t - 1]
        end = splits[t]
        rg = robust_grad(X[start:end, :], y[start:end], w, grad, s, beta)
        if np.any(np.isnan(rg)): print(rg)
        scores = -constraint @ rg
        exp = exponential(scores, 8 * lambd * np.sqrt(2) * s / (3 * (end - start)), epsilon)
        eta = 2 / (t + 2)
        w = (1 - eta) * w + eta * constraint[exp, :]
        w = np.squeeze(np.asarray(w))

    return w