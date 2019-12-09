#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:02:01 2019

@author: kenneth
"""
from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels
from Utils.utils import EvalC
from Utils.Loss import loss
from tau_update import tau
from sklearn.svm import SVC

class libSVM(Kernels, EvalC, loss, tau):
    def __init__(self, kernell = None, C = None):
        '''LibSVM wrapper
        :param:
            :kernel: kernel. default is linear
            :C: penalty. Default is 0.1
        '''
        super().__init__()
        if not kernell:
            kernell = 'linear'
            self.kernell = kernell
        else:
            self.kernell = kernell
        if not C:
            C = .1
            self.C = C
        else:
            self.C = C
        return

    def fit(self, X, y):
        '''
        :param:
            :X: NxD
            :y: Dx1
        '''
        self.model = SVC(kernel = self.kernell, C = self.C).fit(X, y)
        return self
    
    def predict(self, X):
        '''
        :param:
            :X: NxD
        '''
        return self.model.predict(X)
    
    