import numpy as np
from scipy import sparse

def fit(X, y, grad=None, L_0=None, L_1=None, lambd=None, epsilon=None, delta=None):
    """Polyhedral Stochastic Frank-Wolfe Algorithm
    
    Args:
        X: dataset. Bounded L_1 norm. 
        y: targets. Bounded L_1 norm. 
        grad: gradient function. 
        L_0: Lipschitz constant wrt L_1 norm. 
        L_1: Smoothness constant wrt L_1 norm. 
        lambd: L_1 norm bound of feasible set. 
        eps: privacy parameter. 
        delta: privacy parameter.
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, d = X.shape
    r = np.arange(n)
    np.random.shuffle(r)
    X = X[r, :]
    y = y[r]
    B = X[:int(n/2), :]
    B_y = y[:int(n/2)]
    S = X[int(n/2):, :]
    S_y = y[int(n/2):]
    constraint = lambd * sparse.vstack(
        (sparse.identity(X.shape[1], format="csc"), 
         -sparse.identity(X.shape[1], format="csc")),
        format="csc",
    )

    eta = np.log(n / np.log(2 * d)) / n
    x_prev = np.zeros(d)
    d = grad(B, B_y, x_prev)
    d = d.astype(float)
    directional_deriv = constraint.dot(d)
    noised_dd = directional_deriv + \
        np.random.laplace(8 * L_0 * lambd * np.sqrt(np.log(1 / delta)) / \
                          epsilon * np.sqrt(n), size=directional_deriv.shape)
    v = constraint[np.argmin(noised_dd), :]
    x_now = (eta * v).toarray()
    x_now = np.squeeze(np.asarray(x_now))

    for t in range(1, S.shape[0] + 1):
        s = max(
            (1 - eta) ** t * 4 * L_0 * lambd / n, 
            2 * eta * (2 * L_1 * lambd ** 2 + 2 * L_0 * lambd)
        )
        diff = grad(S[t - 1, :], y[t - 1], x_now, single_example=True).astype(float) - \
            grad(S[t - 1, :], y[t - 1], x_prev, single_example=True).astype(float)
        d = (1 - eta) * (d + diff) + \
            eta * grad(S[t - 1, :], y[t - 1], x_now, single_example=True).astype(float)
        directional_deriv = constraint.dot(d)
        noised_dd = directional_deriv + \
            np.random.laplace(2 * s * np.sqrt(n * np.log(1 / delta) / epsilon), size=directional_deriv.shape)
        v = constraint[np.argmin(noised_dd), :]
        v = np.squeeze(np.asarray(v.toarray()))

        x_prev = x_now
        x_now = (1 - eta) * x_prev + eta * v

    return x_now
