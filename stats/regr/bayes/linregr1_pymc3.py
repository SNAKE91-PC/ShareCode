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
size = 50
eps = np.random.normal(size = size, scale = 1)
x = np.arange(0, len(eps))



y = b0 + b1 * x + eps



def b0_bayesianmodel(x, y):
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
        trace = pm.sample(4000, tune = 1000, progressbar=True, chains = 10, cores=4) # draw posterior samples using NUTS sampling
#         print(trace)

        pm.traceplot(trace, var_names = ['b0'])
#         plt.tight_layout();

        return pm.summary(trace, var_names = ['b0'])



def b0b1_bayesianmodel(x, y):
    with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors         
    #     sigma = HalfCauchy('sigma', beta=10, testval=1.)
        sigma = pm.HalfCauchy('sigma', beta = 10)
        intercept = pm.Normal('b0', 10, sd=2)
        x_coeff = pm.Normal('b1', 10, sd=20)
        
        # Define likelihood
        likelihood = pm.Normal('y', mu=intercept + x_coeff * x, 
                            sd=sigma, observed=y)
        
        # Inference!
        trace = pm.sample(4000, tune = 2000, progressbar=True, chains = 10, cores=4) # draw posterior samples using NUTS sampling
#         print(trace)

        pm.traceplot(trace, var_names = ['b0', 'b1', 'sigma'])
#         plt.tight_layout();

        return pm.summary(trace, var_names = ['b0', 'b1', 'sigma'])


def b0b1_independentbayesian(x, y):
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
        trace = pm.sample(4000, tune = 1000, progressbar=True, chains = 10, cores=4) # draw posterior samples using NUTS sampling
        #         print(trace)
        
        pm.traceplot(trace, var_names = ['b0'])
        #         plt.tight_layout();
        
        return pm.summary(trace, var_names = ['b0'])
    
        
    
    


resb0 = b0_bayesianmodel(x, y)
resb0b1 = b0b1_bayesianmodel(x, y)
    
print("")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
