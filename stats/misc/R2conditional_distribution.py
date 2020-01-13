'''
Created on 10 Jan 2020

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

n = 1000

def plots(y, x):
    
    plt.scatter(np.asarray(x[:,0].T), np.asarray(y), s = 2)
    plt.scatter(np.asarray(x[:,1].T), np.asarray(y), s = 2)
    plt.scatter(np.asarray(x[:,2].T), np.asarray(y), s = 2)
    plt.scatter(np.asarray(x[:,3].T), np.asarray(y), s = 2)

    plt.scatter(np.asarray(y - coeff[:,0:3] * x[:,0:3].T), np.asarray(x[:,3]), s = 2 )
    plt.scatter(np.asarray(y - coeff[:,0:2] * x[:,0:2].T - coeff[:,3] * x[:,3].T), np.asarray(x[:,2]), s = 2 )



nregr = 4
coeff = np.asmatrix(np.linspace(0.5, 2, nregr))

nSim = 1000

for i in range(nSim):
    
    x = np.asmatrix(np.random.normal(size = (n, len(coeff) + 1)))
    y = coeff * x[:, 0:-1].T + x[:,-1].T

#     st.linregress(np.asarray(x[:,3].T), np.asarray(y - coeff[:,0:3] * x[:,0:3].T))
#     st.linregress(np.asarray(np.asarray(x[:,2]).T), y - coeff[:,0:2] * x[:,0:2].T - coeff[:,3] * x[:,3].T)
                         

                         
print("")

