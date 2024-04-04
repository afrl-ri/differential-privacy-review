from linear_regression import LinearRegression
import numpy as np
from sklearn.datasets import fetch_openml, load_svmlight_file
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import csv
from functools import partial
from multiprocessing import Pool

from pcoptim import LeastSquares, Logistic
from pcoptim import L1Regularizer, L2Regularizer, ElasticNetRegularizer
from pcoptim import coordinate_descent, gradient_descent, greedy_coordinate_descent


def training_function(tseed, X=None, y=None, alg=None, eps=None, lambd=None, **kwargs):
    X_train, X_valtest, y_train, y_valtest = \
        train_test_split(X, y, train_size=0.6, random_state=tseed)
    X_val, X_test, y_val, y_test = \
        train_test_split(X_valtest, y_valtest, train_size=0.5, random_state=tseed)
    
    n = X_train.shape[0]
    delta = 1 / (n ** 2)
    priv_params = {"epsilon": eps, "delta": delta}
    reg = L1Regularizer(lambd)
    loss = LeastSquares(X_train, y_train, reg)
    w_0 = np.zeros(X.shape[1])

    ret = greedy_coordinate_descent(loss, w_0, 
                                        **priv_params, 
                                        **kwargs, nb_logs=1)
    
    w = ret.final_coef
    lr = LinearRegression('dpight')
    lr.w = w
    y_valpred = lr.predict(X_val)
    y_testpred = lr.predict(X_test)

    valmse = mean_squared_error(y_val, y_valpred)
    testmse = mean_squared_error(y_test, y_testpred)
    testmae = mean_absolute_error(y_test, y_testpred)
    testr2 = r2_score(y_test, y_testpred)

    return valmse, testmse, testmae, testr2


X_bodyfat, y_bodyfat = load_svmlight_file('data/bodyfat')
X_bodyfat = X_bodyfat.toarray()
X_bodyfat = StandardScaler().fit_transform(X_bodyfat)
y_bodyfat = y_bodyfat - np.mean(y_bodyfat)

X_PAH, y_PAH = fetch_openml(data_id=424, data_home='data', return_X_y=True, as_frame=False)
X_PAH = StandardScaler().fit_transform(X_PAH)
y_PAH = y_PAH - np.mean(y_PAH)

X_E2006, y_E2006 = load_svmlight_file('data/E2006.train')
X_E2006 = np.load('data/E2006TSVD.npy')
y_E2006 = y_E2006[:250]
X_E2006 = StandardScaler().fit_transform(X_E2006)
y_E2006 = y_E2006 - np.mean(y_E2006)

datasets = ['bodyfat', 'PAH', 'E2006']
algorithms = ['gcdgss', 'gcdgsr', 'gcdgsq']

params = {
    'gcdgss': {
        'learning_rate': [1e-3, 1e-2, 1e-1], 
        'strategy': ['GSs'], 
        'clip': [1e3], 
        'max_iter': [1, 2, 5, 10, 20, 50, 100], 
        'epochs': [False], 
        'lambd': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    }, 
    'gcdgsr': {
        'learning_rate': [1e-3, 1e-2, 1e-1], 
        'strategy': ['GSr'], 
        'clip': [1e3], 
        'max_iter': [1, 2, 5, 10, 20, 50, 100], 
        'epochs': [False], 
        'lambd': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    }, 
    'gcdgsq': {
        'learning_rate': [1e-3, 1e-2, 1e-1], 
        'strategy': ['GSq'], 
        'clip': [1e3], 
        'max_iter': [1, 2, 5, 10, 20, 50, 100], 
        'epochs': [False], 
        'lambd': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    }
}

epsilons = [0.1, 0.5, 1, 2, 5]
rng = np.random.default_rng(seed=42)
for d in datasets:
    if d == 'bodyfat':
        X, y = X_bodyfat, y_bodyfat
    elif d == 'PAH': 
        X, y = X_PAH, y_PAH
    else:
        X, y = X_E2006, y_E2006

    X = X / np.max(np.linalg.norm(X, ord=1, axis=0))
    X = X / np.max(np.linalg.norm(X, ord=1, axis=1))
    y = y / np.max(np.abs(y))
    
    for a in algorithms:
        param = params[a]

        pgrid = [{**p, "seed": s}
                 for p in ParameterGrid(param)
                 for s in rng.integers(100000, size=2)]

        with open(f'newresults/{d}/{a}.csv', 'a') as f:
            csvwriter = csv.writer(f, lineterminator='\n')
            for e in epsilons:
                for p in pgrid:
                    with Pool(15) as pool:
                        for valmse, testmse, testmae, testr2 in pool.imap_unordered(
                            partial(training_function, X=X, y=y, alg=a, eps=e, **p), 
                            list(range(20))):
                            csvwriter.writerow([
                                d, 
                                a, 
                                e, 
                                valmse, 
                                testmse, 
                                testmae, 
                                testr2
                            ])

                            f.flush()
                
                



