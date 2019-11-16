#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:55:02 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC
from Utils.Loss import loss
from tau_update import tau

class kernelpassiveAggr(EvalC, Kernels, loss, tau):
    def __init__(self, kernel = None, gamma = None, d = None, C = None):
        if not kernel:
            kernel = 'linear'
            self.kernel= kernel
        else:
            self.kernel = kernel
        if not gamma:
            gamma = 8
            self.gamma = gamma
        else:
            self.gamma = gamma
        if not d:
            d = 3
            self.d = d
        else:
            self.d = d
        if not C:
            C = .1
            self.C = C
        else:
            self.C = C
        return
    
    def classictau(self, x, loss):
        '''
        :param: x: NxD
        :param: loss
        '''
        return loss/self.kernelize(x, x)
    
    def f1_relax(self, x, loss, C):
        '''
        :param: x: NxD
        :param: loss
        :param: C: constant
        '''
        return min(C, loss/self.kernelize(x, x))
    
    def f2_relax(self, x, loss, C):
        '''
        :param: x: NxD
        :param: loss
        :param: C: constant
        :return: f2-relaxation
        '''
        return loss/(self.kernelize(x, x) + 2*C)
    
    def kernelize(self, x1, x2):
        '''
        :params: X: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2, gamma = self.gamma)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2, d = self.d)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2, gamma = self.gamma)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2, gamma = self.gamma)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
        
    def activation(self, X, y, alpha):
        '''
        :params: X: train data
        :params: X: weights
        '''
        return 1 if y*(np.dot(alpha, self.kernelize(X, X))) >0 else 0
    
    def cost(self, X, y, alpha):
        '''
        :params: X: N x D
        :params: y: D x 1
        :params: beta: weights D x 1 
        '''
        return y*(np.dot(alpha, self.kernelize(X, X)))
        
    def pred_update(self, X, alpha):
        '''
        :param: X: N x D
        :param: beta: D X 1
        '''
        return np.sign(np.dot(alpha, self.kernelize(X, X)))
    
    def fit(self, X, Y):
        '''
        :params: X: train data
        :params: Y: train labels
        :params: alpha: learning rate
        '''
        self.X = X
        self.alpha = np.random.randn(X.shape[0])
        self.pred = np.zeros(len(Y))
        for ij, (x_i, y_i) in enumerate(zip(self.X, Y)):
            self.pred[ij] = self.pred_update(x_i, self.alpha[ij])
            self.l_t = self.activation(x_i, y_i, self.alpha[ij])
            print(f'Cost of computation: {self.l_t}')
            self.t_t = self.classictau(x_i, self.l_t)
            if self.pred[ij] != y_i:
                self.alpha[ij] = self.alpha[ij] + self.t_t * y_i * self.kernelize(x_i, x_i)
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return np.sign(np.dot(self.alpha, self.kernelize(self.X, X)))
    

class kernelpassiveAggr_v2(EvalC, Kernels, loss, tau):
    def __init__(self, kernel = None, gamma = None, d = None, C = None):
        if not kernel:
            kernel = 'linear'
            self.kernel= kernel
        else:
            self.kernel = kernel
        if not gamma:
            gamma = 5
            self.gamma = gamma
        else:
            self.gamma = gamma
        if not d:
            d = 3
            self.d = d
        else:
            self.d = d
        if not C:
            C = .1
            self.C = C
        else:
            self.C = C
        return
    
    def classictau(self, k, loss):
        '''
        :param: x: NxD
        :param: loss
        '''
        return loss/np.linalg.norm(k)
    
    def f1_relax(self, x, loss, C):
        '''
        :param: x: NxD
        :param: loss
        :param: C: constant
        '''
        return min(C, loss/self.kernelize(x, x))
    
    def f2_relax(self, x, loss, C):
        '''
        :param: x: NxD
        :param: loss
        :param: C: constant
        :return: f2-relaxation
        '''
        return loss/(self.kernelize(x, x) + 2*C)
    
    def kernelize(self, x1, x2):
        '''
        :params: X: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2, gamma = self.gamma)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2, d = self.d)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2, gamma = self.gamma)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2, gamma = self.gamma)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
        
    def activation(self, k, y, alpha):
        '''
        :params: X: train data
        :params: X: weights
        '''
        return 1 if y*(np.dot(alpha, k)) >0 else 0
    
    def cost(self, X, y, alpha):
        '''
        :params: X: N x D
        :params: y: D x 1
        :params: beta: weights D x 1 
        '''
        return y*(np.dot(alpha, self.kernelize(X, X)))
        
    def pred_update(self, X, alpha):
        '''
        :param: X: N x D
        :param: beta: D X 1
        '''
        return np.sign(np.dot(alpha, self.kernelize(X, X)))
    
    def yhat(self, k, alpha):
        '''
        :param: k: N x N
        :param: alpha: N X 1
        '''
        return np.sign(np.dot(alpha, k))
    
    def fit(self, X, Y):
        '''
        :params: X: train data
        :params: Y: train labels
        :params: alpha: learning rate
        '''
        self.X = X
        self.alpha = np.random.randn(X.shape[0])
        self.pred = np.zeros(len(Y))
        self.knl = self.kernelize(self.X, self.X)
        for ij, (x_i, y_i) in enumerate(zip(self.knl, Y)):
            print(f'{self.yhat(x_i, self.alpha)}')
            self.pred[ij] = self.yhat(x_i, self.alpha)
            self.l_t = self.activation(x_i, y_i, self.alpha)
            self.t_t = self.classictau(x_i, self.l_t)
            print(f'Cost of computation: {self.l_t}')
            print(f'tau: {self.t_t}')
            if self.pred[ij] != y_i:
                self.alpha = self.alpha + self.t_t * y_i * x_i
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return np.sign(np.dot(self.alpha, self.kernelize(self.X, X)))
    
    
#%% Testing

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#X, y = make_blobs(n_samples=10000, centers = 2, n_features = 2, random_state=0)
X, y = make_circles(n_samples=10000, factor=.05, noise=0.1)
#X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
psaggr = kernelpassiveAggr(kernel = 'rbf').fit(X_train, Y_train)
#psaggr.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = psaggr.predict(X_test))



