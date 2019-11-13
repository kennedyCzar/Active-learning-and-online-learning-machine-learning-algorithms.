#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:24:12 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC
from Utils.Loss import loss
from tau_update import tau

class passiveAggr(EvalC, Kernels, loss, tau):
    def __init__(self):
        
        return
    
    
    @staticmethod
    def activation(X, y, beta):
        '''
        :params: X: train data
        :params: X: weights
        '''
        return 1 if loss.hinge(X, y, beta) >0 else 0
    
    @staticmethod
    def cost(X, y, beta):
        '''
        :params: X: N x D
        :params: y: D x 1
        :params: beta: weights D x 1 
        '''
        return loss.hinge(X, y, beta)
        
    def pred_update(self, X, beta):
        '''
        :param: X: N x D
        :param: beta: D X 1
        '''
        return np.sign(np.dot(beta, X))
    
    def fit(self, X, Y):
        '''
        :params: X: train data
        :params: Y: train labels
        :params: alpha: learning rate
        '''
        self.beta = np.zeros(X.shape[1])
        self.pred = np.zeros(len(Y))
        for ij, (x_i, y_i) in enumerate(zip(X, Y)):
            self.pred[ij] = self.pred_update(x_i, self.beta)
            self.l_t = passiveAggr.activation(x_i, y_i, self.beta)
            print(f'Cost of computation: {self.l_t}')
            self.t_t = tau.classic(x_i, self.l_t)
            if self.pred[ij] != y_i:
                self.beta = self.beta + self.t_t * y_i * x_i
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return np.sign(np.dot(X, self.beta))



    
    
#%% Testing

from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples=10000, centers=2, n_features=2, random_state=1)
#X = np.c_[np.ones(X.shape[0]), X]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)
psaggr = passiveAggr().fit(X_train, Y_train)
psaggr.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = psaggr.predict(X_test))
