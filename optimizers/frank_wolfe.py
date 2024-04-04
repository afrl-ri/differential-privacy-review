import numpy as np
from scipy import sparse
from optimizers.utils import mse_grad

def fit(X, y, grad=None, lambd=None, T=None, epsilon=None, delta=None):
    """Fit function for Frank-Wolfe.

    Args:
        X: dataset with L_infinity norm of 1
        y: targets, {0, 1} for logistic regression, must be scaled for linear regression
        grad: gradient function. For linear regression, each term of the loss should be 0.5 * MSE. 
              For logistic regression, each term should be BCE. 
        lambd: L_1 constraint of feasible set
        T: number of iterations
        epsilon: privacy paramter
        delta: privacy parameter

    Returns: 
        w: (eps, del)-DP weight vector    
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, p = X.shape
    w = np.zeros(p)

    constraint = lambd * sparse.vstack(
        (sparse.identity(p, format="csc"), -sparse.identity(p, format="csc")),
        format="csc",
    )

    for t in range(1, T + 1):
        g = grad(X, y, w)
        g = g.astype(float)

        directional_deriv = constraint.dot(g)
        if grad == mse_grad:
            directional_deriv = directional_deriv + np.random.laplace(
                        scale=(2 * lambd * np.sqrt(8 * T * np.log(1 / delta))) /
                            (n * epsilon), size=(2 * p))
        else:
            directional_deriv = directional_deriv + np.random.laplace(
                        scale=(lambd * np.sqrt(8 * T * np.log(1 / delta))) /
                            (n * epsilon), size=(2 * p))
        w_tilde = constraint[np.argmin(directional_deriv), :].T
        w_tilde = np.squeeze(w_tilde.toarray())
        w = w + (2 / (t + 2)) * (w_tilde - w)
    return w


