import numpy as np

def hom(p):
    return np.array([p[1],p[0],1.])
def hom_(p):
    return np.array([p[0],p[1],1.])

def skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def matToeuc(p):
    return np.array([p[1], p[0]])