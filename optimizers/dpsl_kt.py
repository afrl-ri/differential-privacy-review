from . import utils
from .dp_ight import _hard_threshold
import numpy as np


def _iterative_gradient_hard_thresholding(X, y, grad, bound, s, T, eta, lambd):
    """Gradient descent with hard thresholding. 
    
    Args:
        X: dataset
        y: features
        grad: gradient function. For linear regression, each term of the loss should be 0.5 * MSE. 
              For logistic regression, each term should be BCE. 
        bound: bound on L_1 norm of w. Only required for linear regression
        s: number of nonzero coefficients
        T: number of iterations
        eta: learning rate
        lambd: multiplicand for L_2 regularization in loss

    Returns:
        w: weight vector
    """
    w = np.random.normal(size=X.shape[1])
    for t in range(T):
        w = w - eta * grad(X, y, w, lambd=lambd)
        w = _hard_threshold(w, s)
        if bound is not None and np.linalg.norm(w, ord=1) > bound: 
            w *= bound / np.linalg.norm(w, ord=1)
    return w


def fit(X, y, grad=None, bound=None, X_tilde=None, s=None, 
        eta_1=None, eta_2=None, T_1=None, T_2=None, 
        gamma=None, lambd=None, beta=None, 
        epsilon=None, delta=None):
    """Fit function for DPSL-KT. 
    
    Args:
        X: dataset
        y: targets, {0, 1} for logistic regression
        grad: gradient function. For linear regression, each term of the loss should be 0.5 * MSE. 
              For logistic regression, each term should be BCE. 
        bound: bound on L_1 norm of w. Only required for linear regression
        X_tilde: generated dataset for student network
        s: number of nonzero coefficients
        eta_1: learning rate for teacher network
        eta_2: learning rate for student network
        T_1: number of iterations for teacher network
        T_2: number of iterations for student network
        gamma: maximum infinity norm of gradient. For linear regression, (K*bound + r)K, where K is 
            L_infinity bound on X and R is absolute bound on y. For logistic regression, K. 
        lambd: multiplicand for L_2 regularization in loss
        beta: L_2 bound on covariance matrix of X_tilde. 1/3 for uniform, 1 for normal. 
        epsilon: privacy parameter
        delta: privacy parameter

    Returns:
        w_priv: (eps, del)-DP weight vector
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    if lambd is None: lambd = 0
    if X_tilde is None: X_tilde = np.random.uniform(low=-1, high=1, size=(10000, X.shape[1]))
    n, p = X.shape
    sigma = np.sqrt(
        8 * X_tilde.shape[0] * beta * s * gamma ** 2 * \
        np.log(2.5 / delta) / \
        (n ** 2 * epsilon ** 2 * lambd ** 2)
    )

    w_hat = _iterative_gradient_hard_thresholding(X, y, grad, bound, s, T_1, eta_1, lambd)
    y_tilde = X_tilde @ w_hat + np.random.normal(scale=sigma, size=X_tilde.shape[0])
    w_priv = _iterative_gradient_hard_thresholding(X_tilde, y_tilde, utils.mse_grad, None, s, T_2, eta_2, 0)
    return w_priv
