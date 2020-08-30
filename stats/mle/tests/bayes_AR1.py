'''
Created on Sep 8, 2019

@author: snake91
'''

import pymc3 as pm

import numpy as np
import matplotlib.pyplot as plt
import mle.simulate as sim


np.random.seed(1)
phi1 = 0.1


y = sim.arpGaussian(t = 200, phi = [phi1])



def b0_bayesianmodel(y):
    with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
    #     sigma = HalfCauchy('sigma', beta=10, testval=1.)
        sigma = 1
        intercept = pm.Uniform('phi1', -0.99, 0.99)
        x_coeff = phi1#Normal('x', 0, sd=20)
        
        # Define likelihood
        likelihood = pm.Normal('y', mu= np.hstack([[0], phi1 * y[1:]]), 
                            sd=sigma, observed=y)
        
        # Inference!
        trace = pm.sample(1000, tune = 500, progressbar=True, chains = 2, cores=4) # draw posterior samples using NUTS sampling
#         print(trace)

        pm.traceplot(trace, var_names = ['phi1'])
        plt.tight_layout();

        return pm.summary(trace, var_names = ['phi1'])



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
        plt.tight_layout();

        return pm.summary(trace, var_names = ['b0', 'b1', 'sigma'])



resb0 = b0_bayesianmodel(y)
resb0b1 = b0b1_bayesianmodel(y)
    
print("")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
