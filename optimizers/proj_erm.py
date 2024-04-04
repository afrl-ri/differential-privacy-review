import cvxpy as cp
from cvxpy.atoms import norm
import numpy as np
from scipy.special import expit
from tqdm import tqdm

def lingrad(x, y, theta, phi):
    return (x.T @ phi.T @ phi @ theta - y) * phi.T @ phi @ x

def loggrad(x, y, theta, phi):
    return (expit(x.T @ phi.T @ phi @ theta) - y) * phi.T @ phi @ x

def fit(X, y, m=None, lambd=None, type='linear', epsilon=None, delta=None):
    """Fit function for Projected ERM. 
    
    Args: 
        X: dataset. L_2 norm should be bounded to 1. 
        y: targets. L_2 norm should be bounded to 1. 
        m: size of the latent space. 
        lambd: L_1 bound on the norm of the weight vector. 
        type: linear or logistic regression. 
        epsilon: privacy parameter. 
        delta: privacy parameter. 

    Returns: 
        theta.value: the theta with minimum L_1 norm that satisfies 
        the constraint calculated. 
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    N = X.shape[0]
    d = X.shape[1]

    phi_tilde = np.random.normal(0, 1, (m, X.shape[1]))
    phi = phi_tilde / np.sqrt(m)

    s = np.linalg.norm(phi.T @ phi, ord=2)
    if type == 'linear':
        L = (s * lambd + 1) * s
    else:
        L = s
    sigma_2 = 32 * L**2 * N**2 * np.log(N / delta) * np.log(1 / delta) / (epsilon**2)
    theta = np.zeros(d)
    for t in range(1, N**2):
        i = np.random.randint(0, high=N)
        if type == 'linear':
            grad = lingrad(X[i, :], y[i], theta, phi)
        else:
            grad = loggrad(X[i, :], y[i], theta, phi)
        b = np.random.multivariate_normal(np.zeros(d), sigma_2 * np.eye(d))
        eta = lambd / np.sqrt(t * (N ** 2 * L ** 2 + d * sigma_2))
        theta = theta - eta * (N * grad + b)
        if np.linalg.norm(theta, ord=1) > lambd:
            theta = theta * lambd / np.linalg.norm(theta, ord=1)
    nu_priv = phi @ theta
    
    theta = cp.Variable(X.shape[1])
    objective = cp.Minimize(norm(theta, 1))
    constraint = [phi @ theta == nu_priv]
    problem = cp.Problem(objective, constraint)
    result = problem.solve()

    return theta.value