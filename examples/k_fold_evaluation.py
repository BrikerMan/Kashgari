#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : BrikerMan
# Site    : https://eliyar.biz

# Time    : 2020/9/3 7:23 下午
# File    : k_fold_evaluation.py
# Project : Kashgari


from sklearn.model_selection import StratifiedKFold
import numpy as np
from kashgari.corpus import SMP2018ECDTCorpus
from kashgari.tasks.classification import BiLSTM_Model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Combine all data for k-folding

train_x, train_y = SMP2018ECDTCorpus.load_data('train')
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
test_x, test_y = SMP2018ECDTCorpus.load_data('test')

X = train_x + valid_x + test_x
Y = train_y + valid_y + test_y

# define 10-fold cross validation test harness
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
scores = []

for train_indexs, test_indexs in k_fold.split(X, Y):
    train_x, train_y = [], []
    test_x, test_y = [], []

    for i in train_indexs:
        train_x.append(X[i])
        train_y.append(Y[i])

    assert len(train_x) == len(train_y)
    for i in test_indexs:
        test_x.append(X[i])
        test_y.append(Y[i])

    assert len(test_x) == len(test_y)
    model = BiLSTM_Model()
    model.fit(train_x, train_y, epochs=10)

    report = model.evaluate(test_x, test_y)
    # extract your target metric from report, for example f1
    scores.append(report['f1-score'])

print(f"{np.mean(scores):.2f}  (+/- {np.std(scores):.2f})")
