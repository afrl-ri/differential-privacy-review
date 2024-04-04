from .dp_ight import _hard_threshold
import cvxpy as cp
from cvxpy.atoms import logistic, norm, sum, sum_squares
import numpy as np
from scipy.special import expit


def mse_grad(X, y, w, lambd=0, per_example=False, single_example=False):
    if per_example: 
        return X * (X @ w - y)[:, None]
    elif single_example:
        return (X * (X @ w - y))
    else:
        return 1 / X.shape[0] * X.T @ (X @ w - y) + lambd * w


def bce_grad(X, y, w, lambd=0, per_example=False, single_example=False):
    try:
        sigmoid_X = expit((X @ w).astype(float))
    except:
        raise Exception(f"{(X @ w).astype(float).dtype}")
    if per_example:
        return X * (sigmoid_X - y)[:, None]
    elif single_example: 
        return X * (sigmoid_X - y)
    else:
        return 1 / X.shape[0] * (sigmoid_X - y).T @ X + lambd * w
    

def mse_support(X, y, s, big_lambd):
    n, p = X.shape
    w = cp.Variable(p)
    error = sum_squares(X @ w - y)
    objective = cp.Minimize(1 / (2 * n) * error + big_lambd / n * norm(w, 1))
    prob = cp.Problem(objective)
    prob.solve()

    w_priv = _hard_threshold(w.value, s)
    return np.where(w_priv != 0, 1, 0)


def bce_support(X, y, s, big_lambd):
    n, p = X.shape
    w = cp.Variable(p)
    bce = -y.T @ (X @ w) + sum(logistic(X @ w))
    objective = cp.Minimize(1 / n * bce + big_lambd / n * norm(w, 1))
    prob = cp.Problem(objective)
    prob.solve()

    w_priv = _hard_threshold(w.value, s)
    return np.where(w_priv != 0, 1, 0)


def mse_obj_pert(X, y, s, xi, small_lambd, big_delta, support, epsilon, delta):
    assert big_delta >= 2 * small_lambd / epsilon
    n, p = X.shape
    if delta == 0: 
        norm = np.random.gamma(shape=p, scale=2 * xi / epsilon)
        direction = np.random.normal(size=p)
        direction /= np.linalg.norm(direction)
        b = norm * direction
    else:
        cov = (xi** 2 * (8 * np.log(2 / delta) + 4 * epsilon) / epsilon ** 2) * np.identity(p)
        b = np.random.multivariate_normal(np.zeros(p), cov)
    
    w = cp.Variable(p)
    error = sum_squares(X @ w - y)
    reg = cp.norm(w, 2) ** 2
    pert = b.T @ w
    objective = cp.Minimize(error / (2 * n) + big_delta / (2 * n) * reg + pert / n)
    constraint = [w[support == 0] == 0]

    prob = cp.Problem(objective, constraint)
    prob.solve()

    return w.value

def bce_obj_pert(X, y, s, xi, small_lambd, big_delta, support, epsilon, delta):
    assert big_delta >= 2 * small_lambd / epsilon
    n, p = X.shape
    if delta == 0: 
        norm = np.random.gamma(shape=p, scale=2 * xi / epsilon)
        direction = np.random.normal(size=p)
        direction /= np.linalg.norm(direction)
        b = norm * direction
    else:
        cov = (xi** 2 * (8 * np.log(2 / delta) + 4 * epsilon) / epsilon ** 2) * np.identity(p)
        b = np.random.multivariate_normal(np.zeros(p), cov)

    w = cp.Variable(p)
    bce = -y.T @ (X @ w) + sum(logistic(X @ w))
    reg = cp.norm(w, 2) ** 2
    pert = b.T @ w
    objective = cp.Minimize(bce / n + big_delta / (2 * n) * reg + pert / n)

    constraint = [w[support == 0] == 0]

    prob = cp.Problem(objective, constraint)
    prob.solve()

    return w.value
    
        

