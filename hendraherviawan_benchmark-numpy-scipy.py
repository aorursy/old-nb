#!/usr/bin/env python
# coding: utf-8



import sys
import timeit

import numpy as np
import scipy as sp
from numpy.distutils.system_info import get_info

print("maxint:  %i\n" % sys.maxsize)




info = get_info('blas_opt')
print('BLAS info:')

for kk, vv in info.items():
    print(' * ' + kk + ' ' + str(vv))




print('numpy version: {}'.format(numpy.__version__))
print(numpy.show_config())




print('scipy version: {}'.format(scipy.__version__))
print(scipy.show_config())




N = int(1e6)
n = 40
A = np.ones((N,n))
C = np.dot(A.T, A)

AT_F = np.ones((n,N), order='F')
AT_C = np.ones((n,N), order='C')




#numpy.dot
print('')
get_ipython().run_line_magic('timeit', 'np.dot(A.T, A)  #')

print('')
get_ipython().run_line_magic('timeit', 'np.dot(AT_F, A)  #')

print('')
get_ipython().run_line_magic('timeit', 'np.dot(AT_C, A)  #')




import scipy.linalg.blas
get_ipython().run_line_magic('timeit', 'scipy.linalg.blas.dgemm(alpha=1.0, a=A.T, b=A.T, trans_b=True)')

get_ipython().run_line_magic('timeit', 'scipy.linalg.blas.dgemm(alpha=1.0, a=A, b=A, trans_a=True)')




X = np.random.random((1000, 3))




# Numpy Function With Broadcasting
def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))
get_ipython().run_line_magic('timeit', 'pairwise_numpy(X)')




# Pure python function
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
get_ipython().run_line_magic('timeit', 'pairwise_python(X)')




# Numba wrapper
from numba import double
from numba.decorators import jit, autojit

pairwise_numba = autojit(pairwise_python)

get_ipython().run_line_magic('timeit', 'pairwise_numba(X)')




#optimize cython function
get_ipython().run_line_magic('load_ext', 'Cython')




get_ipython().run_cell_magic('cython', '', '\nimport numpy as np\ncimport cython\nfrom libc.math cimport sqrt\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef pairwise_cython(double[:, ::1] X):\n    cdef int M = X.shape[0]\n    cdef int N = X.shape[1]\n    cdef double tmp, d\n    cdef double[:, ::1] D = np.empty((M, M), dtype=np.float64)\n    for i in range(M):\n        for j in range(M):\n            d = 0.0\n            for k in range(N):\n                tmp = X[i, k] - X[j, k]\n                d += tmp * tmp\n            D[i, j] = sqrt(d)\n    return np.asarray(D)')




get_ipython().run_line_magic('timeit', 'pairwise_cython(X)')




# scipy pairwise distance
from scipy.spatial.distance import cdist
get_ipython().run_line_magic('timeit', 'cdist(X, X)')




from sklearn.metrics import euclidean_distances
get_ipython().run_line_magic('timeit', 'euclidean_distances(X, X)')






