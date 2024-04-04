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

def training_function(seed, X=None, y=None, alg=None, eps=None, **kwargs):
    X_train, X_valtest, y_train, y_valtest = \
        train_test_split(X, y, train_size=0.6, random_state=seed)
    X_val, X_test, y_val, y_test = \
        train_test_split(X_valtest, y_valtest, train_size=0.5, random_state=seed)
    
    if alg in ['admm', 'admmhalf']:
        y_train[y_train == 0] = -1

    lr = LogisticRegression(alg)
    n = X_train.shape[0]
    delta = 1 / (n ** 2)
    lr.fit(X_train, y_train, epsilon=eps, delta=delta, **kwargs)

    w = lr.w
    yprob_valpred = lr.predict_proba(X_val)
    y_valpred = np.where(yprob_valpred > 0.5, 1, 0) 
    yprob_testpred = lr.predict_proba(X_test)
    y_testpred = np.where(yprob_testpred > 0.5, 1, 0)

    valacc = accuracy_score(y_val, y_valpred)
    testacc = accuracy_score(y_test, y_testpred)
    testbacc = balanced_accuracy_score(y_test, y_testpred)
    testf1 = f1_score(y_test, y_testpred)
    testauroc = roc_auc_score(y_test, yprob_testpred)

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
algorithms = ['admm', 'admmhalf', 'dpight', 'dpslkt', 'fw',
              'htfw', 'htso', 'nm', 'polyfw', 
              'projerm', 'ts', 'vrfw']
params = {
    'admm': {
        'm': [1000], 
        'lr': [0.001], 
        'gamma': [1e-3, 1e-2, 1e-1, 1], 
        'iterations': [1, 2, 5, 10, 20, 50, 100], 
        'l': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    },
    'admmhalf': {
        'm': [1000], 
        'lr': [0.001], 
        'gamma': [1e-3, 1e-2, 1e-1, 1], 
        'T': [100], 
        'iterations': [1, 2, 5, 10, 20, 50, 100], 
        'l': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    },
    'dpight': {
        'bound': [1], 
        'G': [1], 
        's': [1, 2, 5, 10], 
        'eta': [1e-3, 1e-2, 1e-1], 
        'T': [1, 2, 5, 10, 20, 50, 100]
    }, 
    'dpslkt': {
        'bound': [1], 
        'eta_1': [1e-3],
        'T_1': [1000], 
        's': [1, 2, 5, 10], 
        'eta_2': [1e-3, 1e-2, 1e-1], 
        'T_2': [1, 2, 5, 10, 20, 50, 100], 
        'lambd': [1e-3, 1e-2, 1e-1, 1e0], 
        'gamma': [1], 
        'beta': [1/3]
    }, 
    'fw': {
        'lambd': [1], 
        'T': [1, 2, 5, 10, 20, 50, 100]
    }, 
    'htfw': {
        'lambd': [1], 
        'beta': [1], 
        'T': [1, 2, 5, 10, 20, 50, 100], 
        's': [1e0, 1e1, 1e2]
    }, 
    'htso': {
        'T': [1, 2, 5, 10, 20, 50, 100], 
        's': [1, 2, 5, 10], 
        'k': [1e0, 1e1, 1e2], 
        'eta': [1e-3, 1e-2, 1e-1], 
        'beta': [1]
    }, 
    'nm': {
        'G': [1], 
        'lambd': [1]
    }, 
    'polyfw': {
        'L_0': [1], 
        'L_1': [1/4], 
        'lambd': [1]
    }, 
    'projerm': {
        'lambd': [1],
        'm': [2, 5, 10, 20]
    }, 
    'ts': {
        's': [1, 2, 5, 10], 
        'big_lambd': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    }, 
    'vrfw': {
        'D': [1], 
        'L': [1], 
        'beta': [1/4]
    }, 
    'sgd': {
        'batch_size': [32, 64, 128], 
        'lr': [0.001, 0.01, 0.1], 
        'T': [1, 2, 5, 10, 20, 50, 100], 
    }
}

epsilons = [0.1, 0.5, 1, 2, 5]
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
        if d == 'RCV1' and a == 'projerm':
            continue
        param = params[a]
        pgrid = ParameterGrid(param)
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
                
                



