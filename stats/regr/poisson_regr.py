'''
Created on 1 Nov 2019

@author: snake91
'''


import numpy as np
import scipy.stats as st
import scipy.optimize as spo
import matplotlib.pyplot as plt

        
n = 1000
    
lamb = 1

b0 = 2
b1 = 0.5
# X = np.linspace(1, 10, n)
X = np.random.normal(size = n) 

func = np.exp
mu = func(b0 + b1 * X)
Y = np.random.poisson(lam = mu)

def poissonmle(params, y, x, func):
    
    b0, b1 = params
    
#     p = [st.poisson.pmf(y[i], mu = func(b0 + b1 * x[i])) for i in range(len(y))]
    p = st.poisson.pmf(y, mu = func(b0 + b1 * x))
    L = np.log(p)
    L = -np.sum(L)
    
    print(b0, b1, L, sep = '  ')
    return L 


b0 = 4
b1 = 2.
# res = spo.minimize(fun = poissonmle, x0 = (b0, b1), args = (Y, X))
res = spo.differential_evolution(func = poissonmle, bounds = ((-5,5), (-5,5)), args = (Y, X, func))

print(res)

plt.scatter(Y, X)

plt.figure()
plt.scatter(np.log(Y), X)

plt.figure()
plt.hist(Y, bins = 200)
plt.hist(np.random.poisson(lam = np.exp(b0 + b1 * X)), bins = 200)




