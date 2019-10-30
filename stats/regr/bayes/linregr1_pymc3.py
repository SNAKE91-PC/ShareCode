'''
Created on Jul 23, 2019

@author: snake91
'''


import pymc3 as pm

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)
b0 = 5
b1 = 0.1
size = 1000
eps = np.random.normal(size = size, scale = 1)
x = np.arange(0, len(eps))



y = b0 + b1 * x + eps




# size = 100
# true_intercept = 0.2
# true_slope = 0.1
# 
# # x = np.linspace(0, 1, size)
# # y = a + b*x
# true_regression_line = true_intercept + true_slope * np.random.normal(scale=.5, size=size)#x
# # add noise
# y = true_regression_line #+ np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)



with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
#     sigma = HalfCauchy('sigma', beta=10, testval=1.)
    sigma = 10
    intercept = pm.Normal('b0', 10, sd=2)
    x_coeff = b1#Normal('x', 0, sd=20)
    
    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + x_coeff * x, 
                        sd=sigma, observed=y)
    
    # Inference!
    trace = pm.sample(4000, tune = 1000, progressbar=True, chains = 10, cores=1) # draw posterior samples using NUTS sampling
    
    print(trace)
    

# plt.figure(figsize=(7, 7))
print(pm.summary(trace, var_names = ['b0']))
pm.traceplot(trace, var_names = ['b0'])
plt.tight_layout();

    
print("")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
