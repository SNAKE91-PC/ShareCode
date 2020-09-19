'''
Created on 12 Sep 2020

@author: snake91
'''


import numpy as np
import scipy.stats as st
import scipy.optimize as spo
import matplotlib.pyplot as plt

# from skopt import gp_minimize
from itertools import cycle
import pathos.pools as pp

from numba import jit
# np.random.seed(10)


### METHOD OF MOMENTS


def blackscholes_analytic(St, K, r, sigma, T):
    
    sigma = np.array(sigma) if type(sigma) == list else sigma
    
    d1 = (np.log(St/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) )
    d2 = d1 - sigma * np.sqrt(T)
    
    return St * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2) 



def blackscholes_num(St, K, r, sigma, T, scenarios = None):

    sigma = np.array(sigma) if type(sigma) == list else sigma

#     N = 1
    
#     if scenarios is None:
#         x = np.random.normal(size = N, scale = T)
#     else:
    x = scenarios
    
    dSt = r * St + sigma * St * x

    ST = St + dSt

    return np.exp(-r *T) * np.array(list(map(lambda x: np.max([x - K, 0]), ST)))



def msefunc(St, K, r, sigma, T, scenarios, Can):
    
    res = (np.mean(blackscholes_num(St, K, r, sigma, T, scenarios)) - Can)**2
    
#     print(sigma, res)
    return res 
    

def histogram(args):
    
    St, K, r, T, Can, seed = args
    
    np.random.seed(seed)
    scenarios = np.random.normal(size = N, scale = T)
    
    ranges = (slice(10e-2, 0.2, 0.05),)
    res = spo.brute(func = lambda sigma: msefunc(St, K, r, sigma, T, scenarios, Can), ranges = ranges, finish = spo.minimize)[0]
    
#         print(res)
#     Cnum = np.mean(blackscholes_num(St, K, r, res, T, scenarios))
    #     print(res)
    #     print(np.mean(blackscholes_num(St, K, r, res.x, T, scenarios)))
    
#         print((Cnum - Can)**2)

    print(seed)
    
    return res


if __name__ == "__main__":
    
    
    St = 100
    r = 0.01
    sigma = 0.1
    N = 1000
    K = 90
    T = 1
    
    
#     Cnum = blackscholes_num(St, K, r, sigma, T, N)
    Can = blackscholes_analytic(St, K, r, sigma, T)


    scenarios = np.random.normal(size = N, scale = T)
#     Cnum_true = np.mean(blackscholes_num(St, K, r, sigma, T, scenarios))
    
    
    sigmas = np.linspace(10e-4, 0.3, 100)
#     mse = [msefunc(St, K, r, sigma, T, scenarios, Can) for sigma in sigmas]
    
#     plt.plot(sigmas[0:80], mse[0:80])
#     plt.show()
    
    bnds = [(10e-4, 1)]
    x0 = (0.5,)
    
#     res = spo.minimize(fun = lambda sigma: msefunc(St, K, r, sigma, T, scenarios, Can), x0 = x0, bounds = bnds, tol = 10e-16)
    NSIM = 100000
    NSAMPLESIZE = 10000
#     resList = []
    
    pool = pp.ProcessPool(16)
    
#     for i in range(SIM):

    data = [(St, K, r, T, Can, seed) for seed in np.arange(NSIM)]
    resList = pool.map(histogram, data)

    
    
#     res = gp_minimize(lambda x: mse(St, K, r, x, T), 
#                       bnds,
#                       )

    plt.hist(resList, bins = 300)
    plt.show()
    
    print("")
    
    
    
    
    
    
    
    
    
    
    
    