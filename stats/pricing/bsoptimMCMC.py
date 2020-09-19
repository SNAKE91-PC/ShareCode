'''
Created on 13 Sep 2020

@author: snake91
'''



'''
Created on Jul 23, 2019

@author: snake91
'''


import pymc3 as pm

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt



def blackscholes_analytic(St, K, r, sigma, T):
    
    sigma = np.array(sigma) if type(sigma) == list else sigma
    
    d1 = (np.log(St/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) )
    d2 = d1 - sigma * np.sqrt(T)
    
    return St * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2) 


def blackscholes_num(St, K, r, sigma, T, x):

    dSt = r * St + sigma * St * x

    ST = St + dSt

    return np.exp(-r *T) * np.max([ST - K, 0])#np.exp(-r *T) * np.array(list(map(lambda x: np.max([x - K, 0]), ST)))


def msefunc(St, K, r, sigma, T, x, Can):
    
    res = (np.mean(blackscholes_num(St, K, r, sigma, T, x)) - Can)**2
    
#     print(sigma, res)
    return res 



def mcmcsampling(St, K, r, T, Can):
    with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors

        sigma = pm.HalfCauchy('sigma', beta=0.1) # these are the bounds of the optimization
        x = pm.Normal('x', 0.1)
        # Define likelihood
        #intercept + x_coeff * x
        likelihood = pm.HalfCauchy('y', mu = msefunc(St, K, r, sigma, T, x, Can), 
                            sd=10e-8, observed=Can)
        
        # Inference!
        trace = pm.sample(4000, tune = 1000, progressbar=True, chains = 10, cores=4) # draw posterior samples using NUTS sampling
#         print(trace)

        pm.traceplot(trace, var_names = ['sigma'])
#         plt.tight_layout();

        return pm.summary(trace, var_names = ['sigma'])

    
np.random.seed(10)


r = 0.01
St = 100
sigma = 0.05
K = 10
T = 1

Can = blackscholes_analytic(St, K, r, sigma, T)

N = 1000
# x = np.random.normal(size = N)

res = mcmcsampling(St, K, r, T, Can)



    
print("")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
