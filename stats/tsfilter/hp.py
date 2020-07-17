'''
Created on Sep 30, 2019

@author: snake91
'''

import numpy as np
import scipy.stats as st
import scipy.optimize as spo
import scipy.interpolate as sct
import matplotlib.pyplot as plt

n = 100

eps = np.random.normal(size = n)
x = np.linspace(0,10,n)

y = x + eps


def hodrickprescott(signal, lamb):
    
    x = signal[0]
    fsignal = [x]
    
    for idx in range(1, len(signal)-1):
        x = signal[idx] - fsignal[idx]
        fsignal.append(x)
    
    
    return fsignal



        
print("")


