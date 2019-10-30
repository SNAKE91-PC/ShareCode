'''
Created on Oct 12, 2019

@author: snake91
'''


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt 

from collections import Counter

pval = {0: [0.98, 0.01, 0.01], 
        1: [0.01, 0.98, 0.01],
        2: [0.01, 0.01, 0.98]}


t = 5000



def sim_markovchain(t, pmatrix, startvalue):
    
    sample = [startvalue]

    a = np.arange(0, len(pmatrix.keys()))    
    for idx in range(1, t):
        
        value =  np.random.choice(a, size = 1, p = pmatrix[sample[idx-1]])[0]
        sample.append(value)
        
    return sample



def est_markovchain(tseries):

    states = set(tseries)
    transmat = np.zeros(shape = (len(states), len(states)))
    for row in range(1, len(tseries)):
        transmat[tseries[row-1], tseries[row]] += 1 

        
    return transmat / np.sum(transmat, axis = 0) 


sample_a = sim_markovchain(t = t, pmatrix = pval, startvalue = 0)
sample_b = sim_markovchain(t = t, pmatrix = pval, startvalue = 0)

pmatrix_a = est_markovchain(sample_a)
pmatrix_b = est_markovchain(sample_b)


print(pmatrix_a)
print(pmatrix_b)

print("")

 

        
