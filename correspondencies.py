#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import utils as ut
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,corner_kitchen_rosenfeld, canny,
                             plot_matches, BRIEF)





def p_corresponents(foto, foto2):
    
    filtered_coords = corner_peaks(corner_harris(foto), min_distance=5)
    filtered_coords2 = corner_peaks(corner_harris(foto2), min_distance=5)

    extractor = BRIEF()

    extractor.extract(foto, filtered_coords)
    filtered_coords = filtered_coords[extractor.mask]
    descriptors1 = extractor.descriptors

    extractor.extract(foto2, filtered_coords2)
    filtered_coords2 = filtered_coords2[extractor.mask]
    descriptors2 = extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    
    return [(filtered_coords[p1], filtered_coords2[p2]) for p1,p2 in matches12]

def brief(foto, foto2):
    
    keypoints1 = corner_peaks(corner_kitchen_rosenfeld(foto), min_distance=1)
    keypoints2 = corner_peaks(corner_kitchen_rosenfeld(foto2), min_distance=1)

    extractor = BRIEF()

    extractor.extract(foto, keypoints1)
    keypoints1 = keypoints1[extractor.mask]
    descriptors1 = extractor.descriptors

    extractor.extract(foto2, keypoints2)
    keypoints2 = keypoints2[extractor.mask]
    descriptors2 = extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    keypoints1 = np.array([keypoints1[p] for p,q in matches12])
    keypoints2 = np.array([keypoints2[q] for p,q in matches12])
    
    return (keypoints1,keypoints2)

def corr_canny(foto, foto2, F_, K_1, K_2):
    
    im = canny(foto)
    im2 = canny(foto2)
    l1 = []
    l2 = []
    extractor = BRIEF()
    
    u_u = foto.shape[1]/2
    u_z = foto.shape[0]/2
    d_u = foto2.shape[1]/2
    d_z = foto2.shape[0]/2

    sectors = ((0,0), (0,1), (1,0), (1,1))
    
    for a,b in sectors:

        keypoints1 = np.array([np.array([y,x]) for x in range(u_u*a, (1+a)*u_u) for y in range(u_z*b, (1+b)*u_z) if im[y][x]])
        keypoints2 = np.array([np.array([y,x]) for x in range(d_u*a, (1+a)*d_u) for y in range(d_z*b, (1+b)*d_z) if im2[y][x]])

        extractor.extract(foto, keypoints1)
        keypoints1 = keypoints1[extractor.mask]
        descriptors1 = extractor.descriptors

        extractor.extract(foto2, keypoints2)
        keypoints2 = keypoints2[extractor.mask]
        descriptors2 = extractor.descriptors

        matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

        for i1, i2 in matches12:
            p = ut.hom(keypoints1[i1])
            q = ut.hom(keypoints2[i2])
            if abs(np.dot(q, np.dot(F_,p))) <= 1.8:

                l1.append(np.linalg.inv(K_1).dot(p))
                l2.append(np.linalg.inv(K_2).dot(q))
            
    return (l1, l2)

def corr_canny_P(foto, foto2, F_):
    
    im = canny(foto)
    im2 = canny(foto2)
    l1 = []
    l2 = []
    extractor = BRIEF()
    
    u_u = foto.shape[1]/2
    u_z = foto.shape[0]/2
    d_u = foto2.shape[1]/2
    d_z = foto2.shape[0]/2

    sectors = ((0,0), (0,1), (1,0), (1,1))
    
    for a,b in sectors:

        keypoints1 = np.array([np.array([y,x]) for x in range(u_u*a, (1+a)*u_u) for y in range(u_z*b, (1+b)*u_z) if im[y][x]])
        keypoints2 = np.array([np.array([y,x]) for x in range(d_u*a, (1+a)*d_u) for y in range(d_z*b, (1+b)*d_z) if im2[y][x]])

        extractor.extract(foto, keypoints1)
        keypoints1 = keypoints1[extractor.mask]
        descriptors1 = extractor.descriptors

        extractor.extract(foto2, keypoints2)
        keypoints2 = keypoints2[extractor.mask]
        descriptors2 = extractor.descriptors

        matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

        for i1, i2 in matches12:
            p = ut.hom(keypoints1[i1])
            q = ut.hom(keypoints2[i2])
            if abs(np.dot(q, np.dot(F_,p))) <= 1.8:

                l1.append(p)
                l2.append(q)
            
    return (l1, l2)