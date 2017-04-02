#!/usr/bin/python
# -*- coding: utf-8 -*-


from numpy import *


def eight_point_algorithm(x1,x2):
    
    # Normalitzem
    x1, T1 = normalitzacio(x1)
    x2, T2 = normalitzacio(x2)
    
    # Matriu Ax=0
    A = matriu_sistema(x1,x2)
    
    # Obtenim F del vector propi corresponent al menor VAP
    (U, S, V) = linalg.svd(A)
    V = V.conj().T;
    F = V[:,8].reshape(3,3).copy()
    
    # Imposem que tingui rang 2
    (U,D,V) = linalg.svd(F);
    F = dot(dot(U,diag([D[0], D[1], 0])),V)

    # desfem la normalització
    F = dot(dot(T2.T,F),T1)
    return F
    
    
def matriu_sistema(x1,x2):
    npts = x1.shape[1]
    # Matriu del sistema
    A = c_[x2[0]*x1[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], ones((npts,1))]
    return A

def normalitzacio(pts):

    #Considerem punts finits  a aquells que tinguin la tercera coordenada no nul·la
    finiteind = abs(pts[2]) > finfo(float).eps
    pts[0,finiteind] = pts[0,finiteind]/pts[2,finiteind]
    pts[1,finiteind] = pts[1,finiteind]/pts[2,finiteind]
    pts[2,finiteind] = 1
    
    # Centroide dels punts finits
    c = [mean(pts[0,finiteind]), mean(pts[1,finiteind])] 
    
    # Translacio de l'origen al centroide
    newp0 = pts[0,finiteind]-c[0] 
    newp1 = pts[1,finiteind]-c[1] 

    meandist = mean(sqrt(newp0**2 + newp1**2));
    
    scale = sqrt(2)/meandist;

    T = eye(3)
    T[0][0] = scale
    T[1][1] = scale
    T[0][2] = -scale*c[0]
    T[1][2] = -scale*c[1]
    newpts = dot(T, pts)    
    
    return newpts, T

class RansacModel(object):
    
    def __init__(self, debug = False):
        self.debug = debug
    
    def fit(self, data):
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]
        F = eight_point_algorithm(x1, x2)
        return F/F[-1][-1]
    
    def get_error(self, data, F):
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        
        Fx1 = F.dot(x1)
        Fx2 = F.T.dot(x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = (diag(dot(x2.T, dot(F, x1)))) **2 / denom

        return err

def F_from_RANSAC(x1, x2, model = RansacModel(), maxiter = 5000, match_theshold = 1e-3):
    import ransac
    data = vstack((x1.T, x2.T))
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter,match_theshold,30, return_all = True )
    return F, ransac_data['inliers']