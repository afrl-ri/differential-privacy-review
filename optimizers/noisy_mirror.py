import cvxpy as cp
from cvxpy.atoms import abs, norm
import numpy as np
from tqdm import tqdm

def noisy_mirror(X, y, lambd, L, grad, prev, p, n, 
                 eta, init, eps, delta):  
    def h(x, prev, p):
        return (1 / (p - 1)) * norm(x - prev, p) ** 2
    def grad_h(y, prev, p):
        if np.linalg.norm(y - prev, p) == 0:
            return np.zeros(y.shape)
        return (2 / (p - 1)) * (np.abs(y - prev) / \
                                np.linalg.norm(y - prev, p)) ** (p / 2 - 1) * \
            np.abs(y - prev) ** (p / 2) * np.sign(y - prev)
    
    b = int(max(np.sqrt(n / np.log(X.shape[1])), 
            np.sqrt(X.shape[1] / eps)))
    try:
        T = int((n / b) ** 2)
    except: 
        T = 0
    thetas = np.zeros((T + 1, X.shape[1]))
    thetas[0, :] = init

    for k in range(1, T + 1):
        samples = np.random.randint(X.shape[0], size=b)
        S = X[samples, :]
        Y = y[samples]

        sigma = 100 * L * \
            np.sqrt(X.shape[1] * np.log(1 / delta)) / \
            (b * eps)
        g = grad(S, Y, thetas[k - 1, :]) + \
            np.random.normal(loc=0, scale=sigma, size=X.shape[1])
        
        w = thetas[k - 1, :]
        for j in range(1000):
            w = w - 0.001 * (g + 1 / eta * (grad_h(w, prev, p) - grad_h(thetas[k - 1, :], prev, p)))
            w = w.astype(float)
            if np.linalg.norm(w - prev, p) > 2 * L * eta * n * (p - 1):
                xi = w - prev
                xi = xi / np.linalg.norm(xi, p) * 2 * L * eta * n * (p - 1)
                w = prev + xi
        thetas[k, :] = w

        
    if T > 0:
        thetas = 2 / (T * (T + 1)) * \
            thetas * np.arange(T + 1)[:, None]
    return np.sum(thetas, axis=0)


def fit(X, y, G=None, grad=None, lambd=None, epsilon=None, delta=None):
    """Localized Noisy Mirror Descent. 
    
    Args:
        X: dataset. Bounded L_infty norm. 
        y: targets. Must be scaled for linear regression. 
        G: Lipschitz constant with respect to L_1 norm. 
        grad: gradient function. 
        lambd: constraint set L_1 norm. 
        epsilon: privacy parameter. 
        delta: privacy parameter.
    """
    
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    
    k = int(np.ceil(np.log(X.shape[0])))
    p = 1 + 1 / np.log(X.shape[1])
    x = np.zeros(X.shape[1])
    eta = 2 * lambd / G * min(np.sqrt(np.log(X.shape[1])/X.shape[0]), 
                                 epsilon/np.sqrt(X.shape[1] * np.log(X.shape[1]) * np.log(1/delta)))

    for i in range(1, k + 1):
        n_i = 2 ** (-i) * X.shape[0]
        eta_i = 2 ** (-4 * i) * eta

        x = noisy_mirror(X, y, lambd, G, grad, x, p, n_i, eta_i, x, epsilon, delta)
    
    return x
    
