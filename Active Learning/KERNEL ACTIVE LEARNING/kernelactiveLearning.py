#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:54:30 2019

@author: kenneth
"""


from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC
from Utils.Loss import loss
from tau_update import tau

    
class kernelactive(EvalC, Kernels, loss, tau):
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
    
    def binom(self, d, p):
        '''
        '''
        return np.random.binomial(1, (d/(d + p)))
    
    
    def fit(self, X, Y, delta = None):
        '''
        :params: X: train data
        :params: Y: train labels
        :params: alpha: learning rate
        '''
        self.X = X
        if not delta:
            delta = 100
            self.delta = delta
        else:
            self.delta
        self.alpha = np.random.randn(X.shape[0])
        self.pred = np.zeros(len(Y))
        for ij, (x_i, y_i) in enumerate(zip(self.X, Y)):
            self.pred[ij] = self.pred_update(x_i, self.alpha[ij])
            self.z_t = self.binom(self.delta, np.absolute(self.pred[ij]))
            if self.z_t == 1:
                if self.pred[ij] != y_i:
                    self.l_t = self.activation(x_i, y_i, self.alpha[ij])
                    print(f'Cost of computation: {self.l_t}')
                    self.t_t = self.classictau(x_i, self.l_t)
                    self.alpha[ij] = self.alpha[ij] + self.t_t * y_i * self.kernelize(x_i, x_i)
                else:
                    self.alpha[ij] = self.alpha[ij]
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return np.sign(np.dot(self.alpha, self.kernelize(self.X, X)))
    
#%% Testing
        
kactive = kernelactive(kernel = 'linrbf').fit(X_train, Y_train)
#psaggr.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = kactive.predict(X_test))










