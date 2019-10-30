'''
Created on Oct 27, 2019

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt



n_size   = 10
n_series = 1000000
x = [np.random.uniform(size = n_size) for i in range(n_series)]

def onesample_theor_ks(x):

    u = np.linspace(0,1, len(x))
    x = np.sort(x)
    
    diff = np.max(np.abs(x - u))
        

    return diff


def twosample_emp_ks(x, y):
    
    x = np.sort(x)
    y = np.sort(y)

    diff = np.max(np.abs(x-y))
    
    return diff

def nsample_emp_ks(*x):

    return

onesample_theor_statsks = np.array([onesample_theor_ks(x[i]) for i in range(int(len(x)/2)) ])
twosample_emp_statks = np.array([twosample_emp_ks(x[i], x[i-1]) for i in range(1,len(x))])
nsample_emp_statks = np.array([nsample_emp_ks(x[i], x[i-2], x[i-1]) for i in range(2,len(x))])


plt.hist(onesample_theor_statsks, bins = 300, histtype='step')
plt.hist(twosample_emp_statks, bins = 300, histtype='step')
plt.hist(nsample_emp_statks, bins = 300, histtype = 'step')





