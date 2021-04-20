
import numpy as np
cimport numpy as np
from cpython cimport array
from libc.stdio cimport printf
from libc.math cimport fabs
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def map4(double[:] x1, double[:] x2, double[:,:] s, double[:,:] r, double g):
    m = s.shape[0]
    n = s.shape[1]
    with cython.nogil:
        for i in range(m):
            for j in range(n):
                s[i,j] = x1[i] - x2[j]
                r[i,j] = -g * s[i,j]
                s[i,j] = fabs(s[i,j])
                r[i,j] /= (1 + s[i, j])


def covmat(x1, x2, gammav, return_gradhyp=False, return_gradx1=False):
    """Return the covariance between x1 and x2 given parameter gammav."""
    x1 = x1.reshape(1, gammav.shape[0]-1)/np.exp(gammav[:-1]) \
        if x1.ndim < 1.5 else x1/np.exp(gammav[:-1])
    x2 = x2.reshape(1, gammav.shape[0]-1)/np.exp(gammav[:-1]) \
        if x2.ndim < 1.5 else x2/np.exp(gammav[:-1])

    V = np.zeros([x1.shape[0], x2.shape[0]])
    R = np.full((x1.shape[0], x2.shape[0]), 1/(1+np.exp(gammav[-1])))
    S = np.zeros([x1.shape[0], x2.shape[0]])

    if return_gradhyp:
        dR = np.zeros([gammav.shape[0], x1.shape[0], x2.shape[0]])
    elif return_gradx1:
        dR = np.zeros([ x1.shape[1], x1.shape[0], x2.shape[0]])
    for k in range(0, gammav.shape[0]-1):
        if return_gradx1:
            map4(x1[:, k],x2[:, k], S, dR[k], np.exp(-gammav[k]))
        else:
            S = np.abs(np.subtract.outer(x1[:, k], x2[:, k]))
        R *= (1 + S)
        V -= S
        if return_gradhyp:
            dR[k] = (S ** 2) / (1 + S)
    if return_gradhyp or return_gradx1:
        dR = dR.transpose(1,2,0)
    R *= np.exp(V)
    if return_gradhyp:
        dR *= R[:, :, None]
        dR[:, :, -1] = np.exp(gammav[-1]) / ((1 + np.exp(gammav[-1]))) *\
            (1 / (1 + np.exp(gammav[-1])) - R)
    elif return_gradx1:
        dR *= R[:, :, None]
    R += np.exp(gammav[-1])/(1+np.exp(gammav[-1]))
    if return_gradhyp or return_gradx1:
        return R, dR
    else:
        return R