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
    def __init__(self, tau_update = None, C = None):
        '''Online Passive-Aggressive Active learning
        :Reference:
            https://link.springer.com/content/pdf/10.1007%2Fs10994-016-5555-y.pdf
        '''
        if not C:
            C = .1
            self.C = C
        else:
            self.C = C
        if not tau_update:
            tau_update = 'classic'
            self.tau_update = tau_update
        else:
            self.tau_update = tau_update
        return
    
    def updatetau(self, x, loss):
        if self.tau_update == 'classic':
            return tau.classic(x, loss)
        elif self.tau_update == 'f1_relax':
            return tau.f1_relax(x, loss, self.C)
        elif self.tau_update == 'f2_relax':
            return tau.f2_relax(x, loss, self.C)
        
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
            delta = 1
            self.delta = delta
        else:
            self.delta = delta
        self.beta = np.zeros(X.shape[1])
        self.pred = np.zeros(len(Y))
        self.counter = 0
        for ij, (x_i, y_i) in enumerate(zip(X, Y)):
            self.pred[ij] = self.pred_update(x_i, self.beta)
            self.z_t = self.binom(self.delta, np.absolute(self.pred[ij]))
            if self.z_t == 1:
                 if self.pred[ij] != y_i:
                     self.l_t = activelearning.activation(x_i, y_i, self.beta)
#                     print(f'Cost of computation: {self.l_t}')
                     self.t_t = self.updatetau(x_i, self.l_t)
                     self.beta = self.beta + self.t_t * y_i * x_i
                     self.counter +=1
            else:
                self.beta = self.beta
        return self
    
    def predict(self, X):
        '''
        param: X_test = NxD feature matrix
        '''
        return np.sign(np.dot(X, self.beta))
    
    
#%%

#online = activelearning().fit(X_train, Y_train)
#online.predict(X_test)
#plt.scatter(X_test[:, 0], X_test[:, 1], c = online.predict(X_test))
#



