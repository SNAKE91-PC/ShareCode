'''
Created on Sep 2, 2019

@author: snake91
'''


import pymc3 as pm

import numpy as np
import matplotlib.pyplot as plt



b0 = 5
b1 = 0.1
size = 1000
eps = np.random.normal(size = size, scale = 10)
x = np.arange(0, len(eps))

y = b0 + b1 * x + eps


data = dict(x=x, y=y)



with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10)
    intercept = pm.Normal('b0', 10, sd=2)
    x_coeff = pm.Normal('b1', 5, sd=2)
    
    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + b1 * x, 
                        sd=sigma, observed=y)
    
    # Inference!
    trace = pm.sample(2000, tune = 1000, progressbar=True, chains = 2, cores=1) # draw posterior samples using NUTS sampling
    
    print(trace)
    

# plt.figure(figsize=(7, 7))
print(pm.summary(trace, var_names = ['b0', 'b1', 'sigma']))
pm.traceplot(trace, var_names = ['b0', 'b1', 'sigma'])
plt.tight_layout();

    
print("")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
