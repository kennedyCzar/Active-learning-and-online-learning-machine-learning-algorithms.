#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:19:51 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC
from Utils.Loss import loss
from tau_update import tau


class activelearning(EvalC, Kernels, loss, tau):
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
        :param: beta: weights D X 1
        '''
        return np.sign(np.dot(beta, X))
    
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
        if not delta:
            delta = 100
            self.delta = delta
        else:
            self.delta
        self.beta = np.zeros(X.shape[1])
        self.pred = np.zeros(len(Y))
        for ij, (x_i, y_i) in enumerate(zip(X, Y)):
            self.pred[ij] = self.pred_update(x_i, self.beta)
            self.z_t = self.binom(self.delta, np.absolute(self.pred[ij]))
            if self.z_t == 1:
                 if self.pred[ij] != y_i:
                     self.l_t = activelearning.activation(x_i, y_i, self.beta)
                     print(f'Cost of computation: {self.l_t}')
                     self.t_t = tau.classic(x_i, self.l_t)
                     self.beta = self.beta + self.t_t * y_i * x_i
            else:
                self.beta = self.beta
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return np.sign(np.dot(X, self.beta))
    
    
#%%

online = activelearning().fit(X_train, Y_train)
online.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c = online.predict(X_test))




