'''
Created on 10 Jan 2020

@author: snake91
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from functools import partial

for i in range(0,1000):
    print(i)
    
#     x = np.random.normal(size = 1000000)
    x = np.random.uniform(size = 1000000)
    
    f = lambda t, x: np.mean(np.exp(t * x))
    mgf = partial(f, x = x)
    
    t = np.arange(0, 0.1, 0.01)
    
    y = [mgf(t[i]) for i in range(len(t))]
    
#     plt.plot(t, y)
    plt.scatter(t,y, s = 1, color = 'blue')
    
plt.savefig('/home/snake91/git/ShareCode/stats/misc/pics/UniformMGF.svg')




