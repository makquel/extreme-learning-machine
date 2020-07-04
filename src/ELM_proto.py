#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')

import time
import logging
logging.basicConfig(level=logging.DEBUG)

class ELM:
    def __init__(self):
        logging.info(' ELM Class constructor')  
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0,x)

    def tanh(self, x):
        return np.tanh(x)

    def linear(self, x):
        return x

    def sign(self, x):
        return np.sign(x)

    def train(self, Xt, Yd, nh):
        '''
        X_t: Input pattern
        Y_d: Label
        n_h: Hidden nodes
        '''
        # # Fixing random state for reproducibility
        # np.random.seed(7)
        ne = len(Xt[0])
        N = len(Yd)    
        Xt = np.concatenate((Xt, np.ones((N, 1))), axis=1)
        # divide by 10 in order to improve convergence
        W = np.random.rand(ne + 1, nh)/10
        Hi = np.dot(Xt, W)
        H = self.sigmoid(Hi)        
        Bi = np.dot(np.linalg.pinv(H), Yd)
        
        return W, Bi


    def predict(self, Xt, W, B):
        '''
        ELM test unit for prediction
        X_t: test input pattern
        W_i: Weights vector
        B_i: Bias vector
        '''
        N = len(Xt)
        Xt = np.concatenate((Xt, np.ones((N, 1))), axis=1)
        Hi = np.dot(Xt, W)
        H = 0
        H = self.sigmoid(Hi) 
        Y = np.dot(H, B)
        
        return Y        