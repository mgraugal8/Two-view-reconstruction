#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import utils as ut
import cameres as cm   
from skimage.color import rgb2gray
import fonamental as fo
import correspondencies as cor

def main( nom_imatge_1, nom_imatge_2, K_1, K_2):
    
    #Carreguem les fotos del disc
    foto = rgb2gray(np.flipud(plt.imread(nom_imatge_1))[::-1])
    foto2 = rgb2gray(np.flipud(plt.imread(nom_imatge2))[::-1])
    
    #Calculem el primer conjunt de correspondencies
    Rx1, Rx2 = cor.brief(foto, foto2)
    
    #Calculem les matrius fonamentals mitjançant Ransac 
    #(Podríem calcular-ne només una, i l'altre expressar-la com a K_2^-tFK_1^-1)
    Rx1_ = np.array([ut.hom(x) for x in Rx1])
    Rx2_ = np.array([ut.hom(x) for x in Rx2])

    F_, i = fo.F_from_RANSAC(Rx1_, Rx2_)

    Rx1 = np.array([np.linalg.inv(K_1).dot(ut.hom(x)) for x in Rx1])
    Rx2 = np.array([np.linalg.inv(K_2).dot(ut.hom(x)) for x in Rx2])

    F, inliers = fo.F_from_RANSAC(Rx1, Rx2)

    #Guardem les llistes amb les corespondencies
    l1 = []
    l2 = []


    for x in inliers:
        l1.append(ut.hom_(Rx1[x]))
        l2.append(ut.hom_(Rx2[x]))


    a,b = cor.corr_canny(foto, foto2, F_, K_1, K_2)
    l1 = l1 + a
    l2 = l2 + b
    
    x1 = np.array(l1)
    x2 = np.array(l2)
    
    #Calculem les matrius de camera 
    P = cm.camera_calibrada(F)
    P_1 = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
    ind = cm.camera(P_1, P, x1, x2)
    X = np.array([cm.triangular_punt(x1.T[:,j], x2.T[:,j], P_1, P[ind]) for j in range(len(x1))])
    X = cm.borrar(X)
    
    #dibuixem els punts triangulats
    cm.plot(X)