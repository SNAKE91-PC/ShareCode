'''
Created on 24 Jul 2020

@author: snake91
'''

import mle.simulate as sim
import matplotlib.pyplot as plt
import numpy as np
import mle.likelihood as logl
import scipy.optimize as spo
import scipy.stats as st

import emcee

n = 500

from numpy.linalg import inv

flatten = lambda x: [item for sublist in x for item in sublist]


def lnprob_trunc_norm(x, mean, bounds, C):
    if np.any([x[0] < bounds[0][0], x[1] > bounds[0][1]]) or np.any([x[0] < bounds[1][0], x[1] > bounds[1][1]]):
        return -np.inf
    else:
#         print(x)
        return -0.5*(x-mean).dot(inv(C)).dot(x-mean)


    
y = sim.armapqGaussian(t = n, phi = [0.5], psi = [], y0 = [0.])

paramsN = spo.minimize(logl.maxARMApqN, x0 = (0.5,), args = (y, 1, 0))


q = 50

bound = np.percentile(y, q )
x = np.where(y >= bound, np.nan, y)

Nwalkers = 10000
Ndim = 2
C = np.asarray([[0.0001, 0.], [0., 0.0001]])
mean = [bound, bound]
bounds = np.array([[bound, np.inf], [bound, np.inf]])

pos = np.random.multivariate_normal(mean, C, size=Nwalkers)
Nsteps = 100
S = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob_trunc_norm, args = (mean, bounds, C))

pos, prob, state = S.run_mcmc(pos, Nsteps)


xnanidx = np.where(np.isnan(x))

xnan = np.zeros(shape = (1,n))
xnan[0, xnanidx[0]] = 1

c = max(x)

tol = 0.001
while True:

    simcensored = st.truncnorm.ppf(np.random.uniform(size = len(xnanidx)), a= c,b = np.inf)
    
    paramsN = spo.minimize(logl.maxARMApqN, x0 = (0.5,), args = (x, 1, 0))
    
    


plt.plot(x)
plt.show()