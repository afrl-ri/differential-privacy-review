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

def training_function(seed, X=None, y=None, alg=None, eps=None, **kwargs):
    X_train, X_valtest, y_train, y_valtest = \
        train_test_split(X, y, train_size=0.6, random_state=seed)
    X_val, X_test, y_val, y_test = \
        train_test_split(X_valtest, y_valtest, train_size=0.5, random_state=seed)
    
    lr = LinearRegression(alg)
    n = X_train.shape[0]
    delta = 1 / (n ** 2)
    try:
        lr.fit(X_train, y_train, epsilon=eps, delta=delta, **kwargs)

        w = lr.w
        y_valpred = lr.predict(X_val)
        y_testpred = lr.predict(X_test)

        valmse = mean_squared_error(y_val, y_valpred)
        testmse = mean_squared_error(y_test, y_testpred)
        testmae = mean_absolute_error(y_test, y_testpred)
        testr2 = r2_score(y_test, y_testpred)
    except:
        return 4, 4, 2, 0

    return valmse, testmse, testmae, testr2


X_bodyfat, y_bodyfat = load_svmlight_file('data/bodyfat')
X_bodyfat = X_bodyfat.toarray()
X_bodyfat = StandardScaler().fit_transform(X_bodyfat)
y_bodyfat = y_bodyfat - np.mean(y_bodyfat)

X_PAH, y_PAH = fetch_openml(data_id=424, data_home='data', return_X_y=True, as_frame=False)
X_PAH = StandardScaler().fit_transform(X_PAH)
y_PAH = y_PAH - np.mean(y_PAH)

X_E2006, y_E2006 = load_svmlight_file('data/E2006.train')
X_E2006 = np.load('data/E2006PCA.npy')
y_E2006 = y_E2006[:500]
X_E2006 = StandardScaler().fit_transform(X_E2006)
y_E2006 = y_E2006 - np.mean(y_E2006)

datasets = ['bodyfat', 'PAH', 'E2006']
algorithms = ['dpight', 'dpslkt', 'fw',
              'htfw', 'htpl', 'htsl', 'htso', 'nm', 'polyfw', 
              'projerm', 'ts', 'vrfw']
params = {
    'dpight': {
        'bound': [1], 
        'G': [2], 
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
        'gamma': [2], 
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
    'htpl': {
        'lambd': [1], 
        'K': [1], 
        'T': [1, 2, 5, 10, 20, 50, 100]
    }, 
    'htsl': {
        'T': [1, 2, 5, 10, 20, 50, 100], 
        's': [1, 2, 5, 10], 
        'K': [1], 
        'eta': [1e-3, 1e-2, 1e-1]
    }, 
    'htso': {
        'T': [1, 2, 5, 10, 20, 50, 100], 
        's': [1, 2, 5, 10], 
        'k': [1e0, 1e1, 1e2], 
        'eta': [1e-3, 1e-2, 1e-1], 
        'beta': [1]
    }, 
    'nm': {
        'G': [2], 
        'lambd': [1]
    }, 
    'polyfw': {
        'L_0': [2], 
        'L_1': [1], 
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
        'L': [2], 
        'beta': [1]
    }, 
    'sgd': {
        'batch_size': [32, 64, 128], 
        'lr': [0.001, 0.01, 0.1], 
        'T': [1, 2, 5, 10, 20, 50, 100], 
    }
}

epsilons = [0.1, 0.5, 1, 2, 5]
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
        if d == 'E2006' and a == 'projerm':
            continue
        param = params[a]
        pgrid = ParameterGrid(param)
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
                
                



