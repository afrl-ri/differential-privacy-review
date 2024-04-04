from logistic_regression import LogisticRegression
import numpy as np
from sklearn.datasets import fetch_openml, load_svmlight_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
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
    
    y_train[y_train == 0] = -1
    
    n = X_train.shape[0]
    delta = 1 / (n ** 2)
    priv_params = {"epsilon": eps, "delta": delta}
    reg = L1Regularizer(lambd)
    loss = Logistic(X_train, y_train, reg)
    w_0 = np.zeros(X.shape[1])

    ret = greedy_coordinate_descent(loss, w_0, 
                                        **priv_params, 
                                        **kwargs, nb_logs=1)
    
    w = ret.final_coef
    lr = LogisticRegression('dpight')
    lr.w = w
    yprob_valpred = lr.predict_proba(X_val)
    y_valpred = np.where(yprob_valpred > 0.5, 1, 0) 
    yprob_testpred = lr.predict_proba(X_test)
    y_testpred = np.where(yprob_testpred > 0.5, 1, 0)

    valacc = accuracy_score(y_val, y_valpred)
    testacc = accuracy_score(y_test, y_testpred)
    testbacc = balanced_accuracy_score(y_test, y_testpred)
    testf1 = f1_score(y_test, y_testpred)
    testauroc = roc_auc_score(y_test, yprob_testpred)`

    return valacc, testacc, testbacc, testf1, testauroc


X_heart, y_heart = load_svmlight_file('data/heart')
X_heart = X_heart.toarray()
y_heart[y_heart == -1] = 0
y_heart = y_heart.astype(float)
X_heart = StandardScaler().fit_transform(X_heart)

X_dbworld, y_dbworld = fetch_openml(data_id=1563, data_home='data', return_X_y=True, as_frame=False)
y_dbworld[y_dbworld == '1'] = 0
y_dbworld[y_dbworld == '2'] = 1
y_dbworld = y_dbworld.astype(float)
X_dbworld = StandardScaler().fit_transform(X_dbworld)

X_RCV1, y_RCV1 = load_svmlight_file('data/rcv1_train.binary.bz2')
X_RCV1 = np.load('data/RCV1PCA.npy')
y_RCV1 = y_RCV1[:500]
y_RCV1[y_RCV1 == -1] = 0
y_RCV1 = y_RCV1.astype(float)
X_RCV1 = StandardScaler().fit_transform(X_RCV1)

datasets = ['heart', 'DBworld', 'RCV1']
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
    if d == 'heart':
        X, y = X_heart, y_heart
    elif d == 'DBworld': 
        X, y = X_dbworld, y_dbworld
    else:
        X, y = X_RCV1, y_RCV1

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
                        for valacc, testacc, testbacc, testf1, testauroc in pool.imap_unordered(
                            partial(training_function, X=X, y=y, alg=a, eps=e, **p), 
                            list(range(20))):
                            csvwriter.writerow([
                                d, 
                                a, 
                                e, 
                                valacc, 
                                testacc, 
                                testbacc, 
                                testf1, 
                                testauroc
                            ])

                            f.flush()
                
                



