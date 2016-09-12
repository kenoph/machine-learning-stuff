#!/usr/bin/env python

from __future__ import print_function

import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
import pickle

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

x_train = pd.read_csv(os.path.expanduser("~/dataset/X_train.csv"), index_col=0)
y_train = pd.read_csv(os.path.expanduser("~/dataset/Y_train.csv"),
                      index_col=0)['label']

#x_train = x_train.drop([], 1)


def score(params):
    print("Training with params: ", params)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    params['max_depth'] = int(params['max_depth'])

    cv = xgb.cv(
        params=params,
        dtrain=dtrain,
        early_stopping_rounds=6,
        num_boost_round=150,
        nfold=3,
        stratified=True,
        metrics=['logloss'],
        verbose_eval=True
    )

    score = cv.values[:, 0].min()
    print("\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    space = {
        'eta' : hp.quniform('eta', 0.1, 0.45, 0.05),
        'max_depth' : hp.quniform('max_depth', 3, 8, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 2),
        'subsample' : hp.quniform('subsample', 0.5, 1, 0.1),
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.1),
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'nthread' : 8,
        'silent' : 1
    }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=50)
    print(best)


trials = Trials()
optimize(trials)

