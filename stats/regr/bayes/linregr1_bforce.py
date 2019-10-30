'''
Created on Jul 19, 2019

@author: snake91
'''


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

b0 = 0.2
b1 = 0.1
size = 100
x = np.random.normal(size = size, scale = 1)

y = b0 + b1 * x

#### bayes

likelihood = lambda x, loc, scale: st.norm.pdf(x, loc, scale)
b0_prior = lambda x, loc, scale: st.norm.pdf(x, loc, scale)

b0_set = np.arange(0, 2, 0.01)
b0_mu = 1
b0_sigma = 1

floor = lambda x: 10e-14 if x < 10e-14 else x

NNORMPOSTERIOR = []

for b0_param in b0_set:
    L = [likelihood(y[i], b0_param + b1 * x[i] , b0_sigma) for i in range(len(y))]
    L = list(map(lambda x: floor(x), L))
    L = np.log(L)
    logLikelihood = np.sum(L)
    
    b0_pprob = b0_prior(b0_param, b0_mu, b0_sigma)
    b0_pprob = floor(b0_pprob)
    logb0_pprob = np.log(b0_pprob)
    
    lognnormposterior = logLikelihood + logb0_pprob
    
    nnormposterior = np.exp(lognnormposterior)
    
    NNORMPOSTERIOR.append(nnormposterior)
    
evidence = np.sum(NNORMPOSTERIOR)

B0_POSTERIOR = np.array(NNORMPOSTERIOR) / evidence

B0_PRIOR = b0_prior(b0_set, b0_mu, b0_sigma)
B0_PRIOR = B0_PRIOR/np.sum(B0_PRIOR)

plt.plot(b0_set, B0_PRIOR, label = 'B0_PRIOR')
plt.plot(b0_set, B0_POSTERIOR, label = 'B0_POSTERIOR')
plt.xlim(xmax = 0.6)
plt.legend()


print("")
    
    