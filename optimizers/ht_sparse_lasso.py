import numpy as np

def peeling(v, s, eps, delta, lambd):
    S = np.zeros(s) + v.shape[0]
    for i in range(s):
        j = np.abs(v) + np.random.laplace(
            scale=2 * lambd * np.sqrt(3 * s * np.log(1 / delta) / eps), 
            size=v.shape
        )
        j = np.argsort(j)
        k = j.shape[0] - 1
        while j[k] in S: 
            k -= 1
        S[i] = j[k]
    w = np.random.laplace(
            scale=2 * lambd * np.sqrt(3 * s * np.log(1 / delta) / eps), 
            size=v.shape
    )
    S = S.astype(int)
    ret = np.zeros(v.shape)
    ret[S] = v[S] + w[S]
    return ret


def fit(X, y, grad=None, s=None, K=None, T=None, eta=None, epsilon=None, delta=None):
    """Heavy-Tailed Sparse Lasso
    
    Args:
        X: Dataset. Can be L_1 norm bounded to 1. 
        y: targets. Can be L_1 norm bounded to 1. 
        grad: gradient function. 
        s: sparsity parameter. 
        K: Cutoff for data and targets. 
        T: number of iterations. 
        eta: learning rate. 
        eps: privacy parameter. 
        delta: privacy parameter. 
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, d = X.shape
    w = np.zeros(d)
    splits = np.linspace(0, n, T + 1).astype(int)
    X = np.sign(X) * np.minimum(np.abs(X), K)
    y = np.sign(y) * np.minimum(np.abs(y), K)
    for i in range(1, T + 1):
        start = splits[i - 1]
        end = splits[i]
        w = w - eta * grad(X[start:end, :], y[start:end], w)
        w = peeling(w, s, epsilon, delta, 2 * K**2 * eta * (np.sqrt(s) + 1) / (end - start))
        if np.linalg.norm(w) >= 1:
            w = w / np.linalg.norm(w)
    return w