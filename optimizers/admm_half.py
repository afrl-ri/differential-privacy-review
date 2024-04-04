import cvxpy as cp
from cvxpy.atoms import abs, sum, sum_squares
import numpy as np
from optimizers.admm import _generate_noise, _update_V, _update_w


def _update_Z(p, w, V, c, l, Z, T): 
    """Z update rule for L_1/2 regularized ADMM.

    Args:
        p: number of data features
        w: current weight vector
        V: current V vector
        c: hyperparameter c which is computed from epsilon
        l: lambda multiplicand for L_1/2 regularization
        Z: current Z vector
        T: number of iterations to update T

    Returns:
        Z: the updated Z vector
    """
    for t in range(T):
        Z_var = cp.Variable(p)
        objective = cp.Minimize(l * sum(abs(Z_var) / (Z + np.ones(p)) ** 0.5) + 
                                c / 2 * sum_squares(Z_var - w + V / c))
        prob = cp.Problem(objective)
        prob.solve()
        Z = Z_var.value
    return Z



def fit(X, y, l=None, iterations=None, T=None, gamma=None, m=None, lr=None, epsilon=None, delta=None):
    """Fit function for L_1/2 regularized ADMM. 

    Args: 
        X: dataset
        y: targets, in {-1, 1}
        l: lambda multiplicand for L_1/2 regularization
        iterations: number of iterations to run ADMM for
        T: number of iterations to update Z in each ADMM iteration
        gamma: hyperparameter controlling noise and privacy
        m: number of iterations to update w in each ADMM iteration
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
    Z = np.ones(p)
    w = np.ones(p)
    V = np.zeros(p)

    for i in range(iterations):
        Z = _update_Z(p, w, V, c, l, Z, T)
        w = _update_w(n, X, y, Z, w, V, c, m, lr, b)
        V = _update_V(Z, w, V, c)

    return w