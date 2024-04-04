import numpy as np
from scipy import sparse

class TreeNode:
    def __init__(self):
        self.s = None
        self.x = None
        self.v = None
        self.left = None
        self.right = None
        self.parent = None

def build_tree(t):
    root = TreeNode()
    root.s = ''

    now_add = [root]
    while len(now_add) > 0:
        add = now_add.pop()
        add_left = TreeNode()
        add_left.s = add.s + '0'
        add_left.parent = add
        add_right = TreeNode()
        add_right.s = add.s + '1'
        add_right.parent = add
        add.left = add_left
        add.right = add_right
        if len(add_left.s) < t:
            now_add.extend([add_left, add_right])

    return root


def iter(X, y, grad, lambd, noise, b, t, x):
    root = build_tree(t)
    root.x = x
    samples = np.random.randint(X.shape[0], size=int(b))
    S = X[samples, :]
    Y = y[samples]
    root.v = grad(S, Y, x).astype(float)

    constraint = lambd * sparse.vstack(
        (sparse.identity(X.shape[1], format="csc"), 
         -sparse.identity(X.shape[1], format="csc")),
        format="csc",
    )

    path = []
    stack = [root]
    while stack:
        s = stack.pop()
        path.append(s)
        if s.right is not None:
            stack.extend([s.right, s.left])

    for i in range(1, len(path)):
        if path[i].s[-1] == '0':
            path[i].v = path[i].parent.v
            path[i].x = path[i].parent.x
        else:
            samples = np.random.randint(X.shape[0], size=int(2**(-len(path[i].s)) * b))
            S = X[samples, :]
            Y = y[samples]
            path[i].v = path[i].parent.v + grad(S, Y, path[i].x).astype(float) - grad(S, Y, path[i].parent.x).astype(float)
        if len(path[i].s) == t:
            directional_deriv = constraint.dot(path[i].v)
            noised_dd = directional_deriv + np.random.laplace(scale=noise, size=directional_deriv.shape)
            w = constraint[np.argmin(noised_dd), :]
            w = np.squeeze(np.asarray(w.toarray()))

            eta = 2 / (2**(t - 1) + int(path[i].s, 2) + 1)

            next_x = (1 - eta) * path[i].x + eta * w

            if i == len(path) - 1:
                return next_x
            else:
                path[i + 1].x = next_x

def fit(X, y, grad=None, D=None, L=None, beta=None, epsilon=None, delta=None):
    """Varianced-reduced Frank-Wolfe by tree method. 
    
    Args:
        X: dataset with bounded L_1 norm. 
        y: dataset with bounded L_1 norm. 
        grad: gradient function. 
        D: maximum L_1 norm of constraint set. 
        L: Lipschitz constant with respect to L_1 norm. 
        beta: Smoothness constant with respect to L_1 norm. 
        epsilon: privacy parameter. 
        delta: privacy parameter. 
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    
    n, d = X.shape
    x = np.zeros(d)

    if delta == 0:
        b = n / (np.log(n) ** 2)
        beta = max(beta, L * np.log(2 * d) * np.log(n) ** 2 / (n * epsilon * D))
        T = int(0.5 * np.log(b * epsilon * beta * D / (L * np.log(2 * d))))
        for t in range(1, T + 1):
            lambd_t = 2 * L * D * 2 ** t / (b * epsilon)
            x = iter(X, y, grad, D, lambd_t, b, t, x)
    else:
        b = n / (np.log(n) ** 2)
        beta = max(beta, np.sqrt(n) * L * np.log(n / delta) * np.log(2 * d) / (epsilon * D * np.log(n)))
        T = int(2/3 * np.log(b * epsilon * beta * D / (L * np.log(n / delta) * np.log(2 * d))))
        lambd = L * D * 2 ** (T / 2) * np.log(n / delta) / (b * epsilon)
        for t in range(1, T + 1):
            x = iter(X, y, grad, D, lambd, b, t, x)
    return x