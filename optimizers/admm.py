import numpy as np
from scipy.special import expit

def _generate_noise(shape, scale):
    """Generates a random vector from a gamma distribution. 

    Args:
        shape: shape of the gamma distribution for the norm
        scale: scale of the gamma distribution for the norm

    Returns: 
        b: a vector with uniformly random direction and norm 
            chosen from the gamma density
    """
    norm = np.random.gamma(shape, scale=scale)
    b = np.random.randn(shape)
    b = norm * b / np.linalg.norm(b)
    return b


def _update_Z(p, w, V, c, l):
    """Z update rule for L_1 regularized ADMM.

    Args:
        p: number of dataset features
        w: current weight vector
        V: current V vector
        c: hyperparameter c which is computed from epsilon
        l: lambda multiplicand for L_1 regularization
    
    Returns:
        Z: the updated Z vector
    """
    Q = w - V / c
    Z = np.zeros(p)
    Z = np.where(Q > l / c, Q - l / c, Z)
    Z = np.where(Q <= -l / c, Q + l / c, Z)
    return Z


def _update_w(n, X, y, Z, w, V, c, m, lr, b):
    """w update rule for ADMM.

    Args:
        n: number of samples in dataset
        X: dataset
        y: targets
        Z: current Z vector
        w: current w vector
        V: current V vector
        c: hyperparameter c computed from epsilon
        m: number of iterations to update w
        lr: learning rate for w
        b: random noise vector for objective perturbation

    Returns:
        w: an updated w vector
    """
    for i in range(m):
        exp = expit(-(X @ w) * y)
        col = -y * exp
        s = (1 / n) * np.sum(X * col[:, None], axis=0)
        deriv = s - c * (Z - w + V / c) + c * b
        w = w - lr * deriv
    return w


def _update_V(Z, w, V, c):
    """V update rule for ADMM. 

    Args:
        Z: current Z vector
        w: current w vector
        V: current V vector
        c: hyperparameter c computed from epsilon

    Returns:
        V: updated V vector
    """
    return V + c * (Z - w)


def fit(X, y, l=None, iterations=None, gamma=None, m=None, lr=None, epsilon=None, delta=None):
    """Fit function for L_1 regularized ADMM. 

    Args:
        X: dataset
        y: targets, in {-1, 1}
        l: lambda multiplicand for L_1 regularization
        iterations: number of ADMM iterations
        gamma: hyperparameter controlling noise and privacy
        m: number of iterations each time w is updated
        lr: learning rate for w
        epsilon: privacy parameter

    Returns:
        w: an (eps, 0)-DP weight vector
    """
    n, p = X.shape
    
    epsilon_1 = epsilon / iterations
    c = (8 * gamma + 2.8) / (4 * epsilon_1 * n)

    if c < 1 / (2 * n):
        return np.random.randn(p)
    
    if gamma > c * n - 7 / 20:
        return np.random.randn(p)
    
    b = _generate_noise(p, 1 / gamma)
    w = np.random.randn(p)
    V = np.zeros(p)

    for i in range(iterations):
        Z = _update_Z(p, w, V, c, l)
        w = _update_w(n, X, y, Z, w, V, c, m, lr, b)
        V = _update_V(Z, w, V, c)

    return w
