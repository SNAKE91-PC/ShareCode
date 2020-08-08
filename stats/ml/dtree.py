'''
Created on 18 Jul 2020

@author: snake91
'''


import numpy as np
import scipy.optimize as spo


def entropy(sample, cutPoints):
    
    return sample[cutPoints] - sample[np.array(set(np.arange(len(sample))).difference(cutPoints))]
    


xList = []

features = 1

for idx in range(features):
    x = [np.random.binomial(n = 1, p = 0.5) for i in range(100)]
    xList.append(x)

if features != 1:   
    xTuple = list(zip(*xList))
else:
    xTuple = np.array(xList[0])


spo.optimize()


print("")