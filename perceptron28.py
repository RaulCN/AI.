#não está funcionando
#https://gist.githubusercontent.com/mmolinare/37df9567d271e3ef3c567c437a462639/raw/c1fc0a1ff6913b28e883e793d0ba1045a8d61ba1/perceptron.py

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 14:58:40 2016

@author: mmolinare
"""

import numpy as np
from numpy.random import shuffle


def perceptron(X, y, w0=None, std=True, shuffle=True, maxiter=100):
    """
    Perceptron supervised learning of binary classifiers for linearly 
    separable data.
    
    Args
    ----
    X : 2D array, shape=(N,d)
        Input space
    
    y : 1D array, size=N
        Output space
    
    w0 : 1D array, size=d+1, optional
        Weights vector initial guess. Default is draw from zero-mean,
        unit-variance normal distribution
       
    std : bool, optional
        Flag for performing feature scaling via standardization.
        Default is True
       
    shuffle : bool, optional
        Flag for randomly shuffling data at each iteration. Default
        is True
       
    maxiter : int, optional
        Maximum number of iterations. Default is 100
        
    Returns
    -------
    w : 1D array, size=d+1
        Computed weights vector
       
    niter : int
        Number of iterations performed
        
    """  
    N, d = X.shape
    if w0 is None:
        w0 = np.random.randn(d+1)
    if std:
        X = (X-X.mean(axis=0)) / X.std(axis=0)  # feature scaling 
    X = np.c_[X, np.ones(N)]
     
    w = w0.copy()
    niter = 0
    while niter < maxiter:
        if shuffle:
            temp = np.c_[X, y]
            np.random.shuffle(temp)
            X, y = temp[:,:-1], temp[:,-1]
        for x_n, y_n in zip(X, y):
            h = np.sign(w.dot(x_n))
            if h != np.sign(y_n):
                w += y_n*x_n
                niter += 1
                break
        else:
            break   
            
    return w, niter
