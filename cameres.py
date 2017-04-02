#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils as ut

def camera_calibrada(E):
    U,D,V = np.linalg.svd(E)
    if np.linalg.det(U.dot(V)) < 0:
        V = -V
    W = np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]])  
    P_2 = [np.c_[U.dot(W.dot(V)).T, U[:,2].T], 
           np.c_[U.dot(W.dot(V)).T, -U[:,2].T],
           np.c_[U.dot(W.T.dot(V)).T, U[:,2].T],
           np.c_[U.dot(W.T.dot(V)).T, -U[:,2].T]]
    
    return P_2

def triangular_punt(x1, x2, P1, P2):
    M = np.zeros((6,6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2
    U, S, V = np.linalg.svd(M)
    X = V[-1,:4]
    return X/X[3]

def camera(P_1, P, x1, x2):
    ind = 0
    maxres = 0
    for i in range(4):
        X = np.array([triangular_punt(x1.T[:,j], x2.T[:,j], P_1, P[i]) for j in range(len(x1))])

        d1 = np.dot(P_1,X.T)[2]
        d2 = np.dot(P[i],X.T)[2]

        if sum(d1>0)+sum(d2>0) > maxres:
            maxres = sum(d1>0)+sum(d2>0)
            ind = i
            
    return ind

def camera_P(F):
    e = np.linalg.svd(F.T)[-1][-1]
    e_ = ut.skew(e)
    C = np.c_[e_.dot(F).T, e]
    return C
    
def borrar(X):
    
    i = 0
    a_borrar = []
    for x in X:
        if abs(x[0]) > 1.5*abs(X[:,0]).mean() or abs(x[1]) > 1.5*abs(X[:,1]).mean() or abs(x[2]) > 1.5*abs(X[:,2]).mean():
            a_borrar.append(i)
        i+= 1
        
    for bor in a_borrar[::-1]:
        if X.T.shape[1] == 4:
            X = np.delete(X.T,bor,0)
        else:
            X = X.T
            X = np.delete(X.T,bor,0)
            
    return X

def plot(X):
    X = X.T
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(-X[0],X[1],X[2],'k.')
    ax.axis('off')
    plt.show()   
    