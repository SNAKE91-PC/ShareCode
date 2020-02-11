'''
Created on 10 Jan 2020

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm

n = 20 

def plots(y, x):
    
    # use nregr 4
    plt.scatter(np.asarray(x[:,0].T), np.asarray(y), s = 2)
    plt.scatter(np.asarray(x[:,1].T), np.asarray(y), s = 2)
    plt.scatter(np.asarray(x[:,2].T), np.asarray(y), s = 2)
    plt.scatter(np.asarray(x[:,3].T), np.asarray(y), s = 2)

    plt.scatter(np.asarray(y - coeff[:,0:3] * x[:,0:3].T), np.asarray(x[:,3]), s = 2 )
    plt.scatter(np.asarray(y - coeff[:,0:2] * x[:,0:2].T - coeff[:,3] * x[:,3].T), np.asarray(x[:,2]), s = 2 )



nregr = 2
coeff = np.asmatrix(np.linspace(0.5, 2, nregr))

nSim = 1000
sigma = 5
results = []

for i in range(nSim):
    
    x = np.asmatrix(np.random.normal(size = (n, coeff.shape[1] + 1), scale = sigma))
    y = x[:, 0:-1] * coeff.T + x[:,-1]
    
    model = sm.OLS(y,x[:, 0:-1])
    res = model.fit()
    
    results.append(res)

betas = np.array(list(map(lambda x: x.params, results)))
covs = np.array(list(map(lambda x: x.cov_params(), results))) 
plt.scatter(betas[:,0], betas[:,1])


#     st.linregress(np.asarray(x[:,3].T), np.asarray(y - coeff[:,0:3] * x[:,0:3].T))
#     st.linregress(np.asarray(np.asarray(x[:,2]).T), y - coeff[:,0:2] * x[:,0:2].T - coeff[:,3] * x[:,3].T)
                         

                         
print("")

