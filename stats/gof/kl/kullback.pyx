'''
Created on 22 Dec 2019

@author: snake91
'''

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

import scipy.stats as st
import matplotlib.pyplot as plt

cdef vector[double] minmax(double i, dict a):
    cdef double minmax 
    cdef vector[double] out
     
    try:
        minmax= min(list(filter(lambda x: x > i, a.keys())))
    except ValueError:
        minmax = min(a.keys())
         
    cdef double maxmin
     
    try:
        maxmin = max(list(filter(lambda x: x < i, a.keys())))
    except ValueError:
        maxmin = max(a.keys())
     
    out.push_back(minmax)
    out.push_back(maxmin)
    
    return out
     
def KullbackLeibler(args):
     
    cdef np.ndarray[np.double_t, ndim = 1] psample = args[0]
    cdef np.ndarray[np.double_t, ndim = 1] qsample = args[1]
    cdef int n = <np.int_t>(args[2])
      
    a = plt.hist(psample, bins = n)
  
    cdef np.ndarray[np.double_t, ndim = 1] ax = a[1]
     
    cdef np.ndarray[np.double_t, ndim = 1] ay = a[0]
     
    b = plt.hist(qsample, bins = ax)
 
    adict = dict(zip(ax, ay))    
    ax = ax[:-1]
     
    cdef np.ndarray[np.double_t, ndim = 1] bx = b[1]
    cdef np.ndarray[np.double_t, ndim = 1] by = b[0]
    bdict = dict(zip(bx, by))
    
    bx = bx[:-1]
     
    cdef vector[double] kl
     
    cdef int N = np.sum(ay)
     
    cdef int i
    cdef double p_minmax, p_maxmin, q_minmax, q_maxmin
    cdef double KL
    
    for i in range(len(psample)):
         
        ptmp = minmax(psample[i], adict)
         
        p_minmax = ptmp[0]
        p_maxmin = ptmp[1]
 
        qtmp = minmax(psample[i], bdict)
        q_minmax = qtmp[0]
        q_maxmin = qtmp[1]
         
        pdensity = adict[p_maxmin]/ N
        qdensity = np.max([bdict[q_maxmin]/ N, 10e-20])
         
        KL = pdensity * np.log(pdensity/qdensity)
 
        kl.push_back(KL)
     
    cdef double res = np.sum(kl)

    del args, psample, qsample, ax, ay, bx, by, adict, bdict
    
    plt.close()
    return res 











