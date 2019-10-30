'''
Created on Dec 8, 2018

@author: snake91
'''

import numpy as np
import scipy.stats as st

def clayton(theta, size = 1000):
    
    u = np.random.uniform(size = size)
    v = np.random.uniform(size = size)
    
    y = []
    for i,j in zip(u,v):
        z = max( i**(-theta) * (j**(-theta/(1+theta))-1) + 1, 0)**(-1./theta)
        y.append(z)
        
    y = np.array(y)
    return u, y


    
    


def gaussian(rho, size = 1000):
    
    u = np.random.normal(size = size)
    v = np.random.normal(size = size)
    
    y = []
    for i,j  in zip(u,v):
        z = rho * i + np.sqrt(1-rho**2) * j
        y.append(st.norm.cdf(z))
        
        
    y = np.array(y)
    
    return u,y


    
    
    
    