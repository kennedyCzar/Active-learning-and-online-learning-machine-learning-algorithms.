#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:40:33 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np

class tau:
    def __init__(self):
        return
    
    @staticmethod
    def classic(X, loss):
        '''
        '''
        return loss/np.linalg.norm(X)
    
    def f1_relax(X, loss, C):
        '''
        '''
        if not C:
            C = .1
        else:
            C = C
        return min(C, loss/np.linalg.norm(X))
    
    def f2_relax(X, loss, C):
        '''
        '''
        if not C:
            C = .1
        else:
            C = C
        return loss/(np.linalg.norm(X) + 2*C)
