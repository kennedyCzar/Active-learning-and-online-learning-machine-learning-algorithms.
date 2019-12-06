#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:15:52 2019

@author: kenneth
"""
import os
import time
import pandas as pd
import numpy as np
from onlinepassive import passiveAggr
from activelearning import activelearning
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from libsvm import libSVM
path = '/home/kenneth/Documents/MLDM M2/ADVANCE_ML/Active-learning-and-online-learning-machine-learning-algorithms./DATASET'
color = 'coolwarm_r'

def removeInt(df):
    '''
    :params: 
        df: NxD
    :Return type: dataframe of floats only
    '''
    df = df.copy(deep = True)
    for ii in df.columns:
        if df[ii].dtype == np.int64:
            df = df.drop(ii, axis = 1)
    return df

#%%
train_x = pd.read_csv(os.path.join(path, 'ijcnn1'), sep=' |:', engine='python', header = None, index_col=False)
train_y = train_x.iloc[:, 0]
test_x = pd.read_csv(os.path.join(path, 'ijcnn1.t'), sep=' |:', engine='python', header = None, index_col=False)
test_y = test_x.iloc[:, 0]
val_x = pd.read_csv(os.path.join(path, 'ijcnn1.val'), sep=' |:', engine='python', header = None, index_col=False)
val_y = val_x.iloc[:, 0]
#%%
train_x = removeInt(train_x)
test_x = removeInt(test_x)
val_x = removeInt(val_x).iloc[:, 1:]

#combine train and val
combine = pd.concat([train_x, val_x], axis = 0)
combine.index = np.arange(combine.shape[0])
y = pd.concat([train_y, val_y], axis = 0)
y.index = np.arange(y.shape[0])
combine = np.array(combine)
y = np.array(y)
#%%
plt.scatter(train_x.iloc[:, 0], train_x.iloc[:, 1], s = 1, c = train_y, cmap = 'coolwarm_r')
plt.scatter(test_x.iloc[:, 0], test_x.iloc[:, 1], s = 1, c = test_y, cmap = 'coolwarm_r')
plt.scatter(val_x.iloc[:, 0], val_x.iloc[:, 1], s = 1, c = val_y, cmap = 'coolwarm_r')
#combined dataset
plt.scatter(combine[:, 0], combine[:, 1], s = 1, c = y, cmap = 'coolwarm_r')
#%% Training
start = time.time() #start time
psaggr = passiveAggr(tau_update = 'classic').fit(np.array(combine), y)
print(f'Completes process in {round(time.time() - start, 2)}secs')
pred = psaggr.predict(test_x)
#plt.scatter(test_x.iloc[:, 0], test_x.iloc[:, 1], c = pred, s = 1, cmap = color)
psaggr.summary(test_y, pred)
psaggr.confusionMatrix(test_y, pred)

#%%
start = time.time() #start time
online = activelearning(tau_update = 'f1_relax').fit(np.array(combine), y, delta = 1)
print(f'Completes process in {round(time.time() - start, 2)}secs')
apred = online.predict(test_x)
online.summary(test_y, apred)
online.confusionMatrix(test_y, pred)
plt.scatter(test_x.iloc[:, 0], test_x.iloc[:, 1], c = apred, s = 1, cmap = 'coolwarm_r')


#%% SVM
linsvm = libSVM(kernell = 'linear', C = .1).fit(np.array(train_x), train_y)
#linsvm = SVC(kernel = 'linear', C = .1).fit(np.array(train_x), train_y)
svmpred = linsvm.predict(np.array(test_x))
linsvm.summary(test_y, svmpred)
plt.scatter(test_x.iloc[:, 0], test_x.iloc[:, 1], c = apred, s = 1, cmap = 'coolwarm_r')

#%% Cross validation
score = []
cv = KFold(n_splits = 10)
for tr, te in cv.split(combine, y):
    x_train, x_test, y_train, y_test = combine[tr], combine[te], y[tr], y[te]
    model = activelearning(tau_update = 'f2_relax').fit(np.array(x_train), y_train)
    score.append(model.f1(y_test, model.predict(x_test)))
print(f'Average f1-score: {np.mean(score)}')

#%% Putting it all together

C = [0.1, 1, 10]
ttau = ['classic', 'f1_relax', 'f2_relax']

algo = ['PA', 'APA', 'LIBSVM']


        
result = {'classic': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                              'f1_relax': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                              'f2_relax': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []}, 
                              'libsvm': {'time': [], 'acc': [], 'prec': [], 'rec': [], 'f1': [], 'aurroc': []},
                             }

for alg in algo:
    for c in C:
        if alg == 'LIBSVM':
            print(f'Learning on SVM using {c}')
            start = time.time()
            linsvm = libSVM(kernell = 'linear', C = c).fit(np.array(combine), y)
            end = time.time() - start
            pred = linsvm.predict(test_x)
            result['libsvm']['time'].append(end)
            result['libsvm']['acc'].append(linsvm.accuracy(test_y, pred))
            result['libsvm']['prec'].append(linsvm.precision(test_y, pred))
            result['libsvm']['rec'].append(linsvm.recall(test_y, pred))
            result['libsvm']['f1'].append(linsvm.f1(test_y, pred))
            result['libsvm']['aurroc'].append(roc_auc_score(test_y, pred))
        else:
            for tt in ttau:
                if alg == 'PA':
                    if tt == 'classic':
                        print(f'Running {alg} with {tt} and {c}')
                        start = time.time()
                        psaggr = passiveAggr(tau_update = tt).fit(np.array(combine), y)
                        end = time.time() - start
                        pred = psaggr.predict(test_x)
                        result[tt]['time'].append(end)
                        result[tt]['acc'].append(psaggr.accuracy(test_y, pred))
                        result[tt]['prec'].append(psaggr.precision(test_y, pred))
                        result[tt]['rec'].append(psaggr.recall(test_y, pred))
                        result[tt]['f1'].append(psaggr.f1(test_y, pred))
                        result[tt]['aurroc'].append(roc_auc_score(test_y, pred))
                    else:
                        print(f'Running {alg} with {tt} and {c}')
                        start = time.time()
                        psaggr = passiveAggr(tau_update = tt, C = c).fit(np.array(combine), y)
                        end = time.time() - start
                        pred = psaggr.predict(test_x)
                        result[tt]['time'].append(end)
                        result[tt]['acc'].append(psaggr.accuracy(test_y, pred))
                        result[tt]['prec'].append(psaggr.precision(test_y, pred))
                        result[tt]['rec'].append(psaggr.recall(test_y, pred))
                        result[tt]['f1'].append(psaggr.f1(test_y, pred))
                        result[tt]['aurroc'].append(roc_auc_score(test_y, pred))
                elif alg == 'APA':
                    if tt == 'classic':
                        print(f'Running {alg} with {tt} and {c}')
                        start = time.time()
                        aonline = activelearning(tau_update = tt).fit(np.array(combine), y)
                        end = time.time() - start
                        apred = aonline.predict(test_x)
                        result[tt]['time'].append(end)
                        result[tt]['acc'].append(aonline.accuracy(test_y, apred))
                        result[tt]['prec'].append(aonline.precision(test_y, apred))
                        result[tt]['rec'].append(aonline.recall(test_y, apred))
                        result[tt]['f1'].append(aonline.f1(test_y, apred))
                        result[tt]['aurroc'].append(roc_auc_score(test_y, apred))
                    else:
                        print(f'Running {alg} with {tt} and {c}')
                        start = time.time()
                        aonline = activelearning(tau_update = tt, C = c).fit(np.array(combine), y)
                        end = time.time() - start
                        apred = aonline.predict(test_x)
                        result[tt]['time'].append(end)
                        result[tt]['acc'].append(aonline.accuracy(test_y, apred))
                        result[tt]['prec'].append(aonline.precision(test_y, apred))
                        result[tt]['rec'].append(aonline.recall(test_y, apred))
                        result[tt]['f1'].append(aonline.f1(test_y, apred))
                        result[tt]['aurroc'].append(roc_auc_score(test_y, apred))
                else:
                    pass
        
        

#%%




















